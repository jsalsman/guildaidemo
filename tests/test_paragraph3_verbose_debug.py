import json
import os
from pathlib import Path

from pocketsphinx import Decoder, get_model_path

from app import (
    PARAGRAPHS,
    PRONUNCIATIONS,
    align_phonemes,
    deepgram_transcribe,
    infer_stress_from_word,
    needleman_wunsch_alignment,
    normalize_token,
    read_wav_bytes,
    slice_word_pcm,
)

AUDIO_PATH = Path(__file__).resolve().parent / "abc340c7-fc39-41f0-b1a6-3557f83b7707.wav"


def _decode_with_pocketsphinx(wav_path: Path) -> list[dict[str, float | str]]:
    model_path = get_model_path()
    decoder = Decoder(
        hmm=model_path + "/en-us/en-us",
        dict=model_path + "/en-us/cmudict-en-us.dict",
        lm=model_path + "/en-us/en-us.lm.bin",
        loglevel="FATAL",
    )
    decoder.start_utt()
    with wav_path.open("rb") as f:
        f.read(44)
        while True:
            buf = f.read(2048)
            if not buf:
                break
            decoder.process_raw(buf, False, False)
    decoder.end_utt()

    out = []
    for seg in decoder.seg():
        if seg.word.startswith("<"):
            continue
        out.append({
            "word": seg.word,
            "start": round(seg.start_frame * 0.01, 2),
            "end": round(seg.end_frame * 0.01, 2),
        })
    return out


def test_paragraph3_verbose_debug_output():
    assert AUDIO_PATH.exists(), f"Missing debug audio: {AUDIO_PATH}"

    paragraph = PARAGRAPHS[2]
    wav_bytes = AUDIO_PATH.read_bytes()
    wav_data = read_wav_bytes(wav_bytes)
    duration = len(wav_data.pcm_bytes) / (wav_data.sample_rate * wav_data.channels * wav_data.sample_width)

    print("=== Paragraph 3 prompt ===")
    print(paragraph["display_text"])
    print("=== Audio format ===")
    print(json.dumps({
        "sample_rate": wav_data.sample_rate,
        "channels": wav_data.channels,
        "sample_width": wav_data.sample_width,
        "duration_sec": round(duration, 3),
    }, indent=2))

    print("=== Deepgram words ===")
    dg_words: list[dict] = []
    if os.environ.get("DEEPGRAM_API_KEY", ""):
        try:
            dg_words = deepgram_transcribe(wav_bytes)
            print(json.dumps(dg_words, indent=2))
        except Exception as exc:  # debug test should continue printing other diagnostics
            print(f"Deepgram request failed: {exc}")
    else:
        print("SKIPPED: DEEPGRAM_API_KEY is not set in this environment")

    ps_words = _decode_with_pocketsphinx(AUDIO_PATH)
    print("=== PocketSphinx full-utterance words ===")
    print(json.dumps(ps_words, indent=2))

    print("=== Target pronunciation inventory (stressless CMUDICT phones) ===")
    for target in paragraph["targets"]:
        word = normalize_token(target["word"])
        print(f"{word}: {PRONUNCIATIONS.get(word, [])}")

    print("=== Deepgram alignment and per-target forced alignment ===")
    if not dg_words:
        print("SKIPPED: no Deepgram words available")
    else:
        ref_norm = [normalize_token(t) for t in paragraph["tokens"]]
        hyp_norm = [normalize_token(w["word"]) for w in dg_words]
        mapping = needleman_wunsch_alignment(ref_norm, hyp_norm)

        rows = []
        for target in paragraph["targets"]:
            ref_idx = target["token_index"]
            mapped_idx = mapping.get(ref_idx)
            row = {
                "target": target["word"],
                "expected_stress": target["expected_stress"],
                "ref_idx": ref_idx,
                "mapped_idx": mapped_idx,
            }
            if mapped_idx is None:
                row["status"] = "missing"
                rows.append(row)
                continue

            dg = dg_words[mapped_idx]
            row["deepgram_word"] = dg["word"]
            row["deepgram_confidence"] = round(float(dg["confidence"]), 4)
            row["timing"] = [round(float(dg["start"]), 3), round(float(dg["end"]), 3)]

            segment = slice_word_pcm(wav_data, float(dg["start"]), float(dg["end"]), duration)
            phones = align_phonemes(target["word"], segment, float(dg["start"]))
            inferred = infer_stress_from_word(target["word"], phones)
            row["aligned_phones"] = [p["phone"] for p in phones]
            row["core_phones"] = inferred["core_phones"]
            row["core_durations"] = inferred["core_durations"]
            row["inferred_stress"] = inferred["inferred_stress"]
            row["status"] = "ok" if inferred["inferred_stress"] is not None else "unaligned"
            rows.append(row)

        print(json.dumps(rows, indent=2))

    assert PRONUNCIATIONS.get("content")
