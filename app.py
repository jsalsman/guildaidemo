"""Flask application for evaluating noun/verb syllable stress from WAV audio.

The service accepts paragraph context with marked target words and aligns ASR
output and phone-level timing to infer whether stress was placed on syllable 1
or syllable 2 for each target.
"""

import base64
import hashlib
import io
import json
import math
import os
import re
import uuid
import wave
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any

import cmudict
import requests
from flask import Flask, Response, g, jsonify, render_template, request


DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
DEEPGRAM_URL = (
    "https://api.deepgram.com/v1/listen"
    "?model=nova-2&language=en-US&punctuate=false&diarize=false"
)
VOWEL_BASES = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"
}
MAX_WORD_PAD_SEC = 0.12
BUCKET_DIR = os.environ.get("BUCKET_DIR", "/bucket")
HST_TZ = ZoneInfo("Pacific/Honolulu")
RATIO_EPS = 1e-4


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


_DECODER = None


def log(message: str) -> None:
    """Print request-scoped logs with a request id for easier traceability."""
    rid = getattr(g, "request_id", "-")
    print(f"[request_id={rid}] {message}", flush=True)


def normalize_token(token: str) -> str:
    """Normalize a token to alphanumerics for matching/alignment."""
    token = token.lower().strip()
    token = re.sub(r"['’]", "", token)
    token = re.sub(r"[^a-z0-9]+", "", token)
    return token


def confidence_cubed(confidence: float | None) -> float | None:
    """Re-weight confidence by cubing, emphasizing high-confidence words."""
    if confidence is None:
        return None
    c = max(0.0, min(1.0, float(confidence)))
    return c ** 3


def normalize_bg(value: float | None) -> float:
    """Clamp background intensity values to [0, 1] for rendering."""
    if value is None:
        return 0.0
    return max(0.0, min(1.0, value))


def parse_bool(value: Any) -> bool:
    """Parse a boolean-ish value from form/json payloads."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _ratio_metrics(d1: float | None, d2: float | None) -> tuple[float | None, float | None]:
    """Compute duration ratio features for threshold aggregation."""
    if d1 is None or d2 is None:
        return None, None
    if d2 <= 0:
        return None, None
    ratio = d1 / d2
    ratio_log = math.log((d1 + RATIO_EPS) / (d2 + RATIO_EPS))
    return round(ratio, 6), round(ratio_log, 6)


def _recording_id_hst() -> tuple[str, str]:
    """Build a microsecond recording id and ISO timestamp in HST."""
    now_hst = datetime.now(HST_TZ)
    return now_hst.strftime("%y%m%d%H%M%S%f"), now_hst.isoformat(timespec="microseconds")


def persist_submission(
    *,
    wav_bytes: bytes,
    paragraph: dict[str, Any],
    paragraph_id: int | None,
    analysis: dict[str, Any],
    native_exemplar: bool,
    source: str,
) -> dict[str, Any]:
    """Persist WAV and sidecar JSON into the mounted bucket directory."""
    recording_id, created_at_hst = _recording_id_hst()
    os.makedirs(BUCKET_DIR, exist_ok=True)
    wav_path = os.path.join(BUCKET_DIR, f"{recording_id}.wav")
    json_path = os.path.join(BUCKET_DIR, f"{recording_id}.json")
    tmp_json_path = f"{json_path}.tmp"

    with open(wav_path, "wb") as wf:
        wf.write(wav_bytes)
        wf.flush()
        os.fsync(wf.fileno())

    paragraph_text = paragraph.get("display_text", "")
    paragraph_hash = hashlib.sha256(paragraph_text.encode("utf-8")).hexdigest() if paragraph_text else None
    targets = []
    for target in analysis.get("targets", []):
        d1 = target.get("core_durations", {}).get("syll1")
        d2 = target.get("core_durations", {}).get("syll2")
        ratio, ratio_log = _ratio_metrics(d1, d2)
        targets.append(
            {
                "token_index": target.get("token_index"),
                "word_display": target.get("word"),
                "word_norm": normalize_token(target.get("word", "")),
                "label": target.get("label"),
                "expected_stress": target.get("expected_stress"),
                "inferred_stress": target.get("inferred_stress"),
                "status": target.get("status"),
                "correct": target.get("correct"),
                "core_phones": target.get("core_phones"),
                "core_durations": target.get("core_durations"),
                "duration_ratio": ratio,
                "duration_ratio_log": ratio_log,
                "deepgram_word_index": target.get("deepgram_word_index"),
                "deepgram_confidence": target.get("deepgram_confidence"),
                "deepgram_confidence_cubed": target.get("deepgram_confidence_cubed"),
                "feedback": target.get("feedback"),
            }
        )

    sidecar = {
        "schema_version": 1,
        "recording_id": recording_id,
        "created_at_hst": created_at_hst,
        "timezone": "Pacific/Honolulu",
        "source": source,
        "request_id": getattr(g, "request_id", None),
        "request_ip": getattr(g, "client_ip", None),
        "paragraph_id": paragraph_id,
        "paragraph_text_hash": f"sha256:{paragraph_hash}" if paragraph_hash else None,
        "native_exemplar": native_exemplar,
        "audio": {
            "path": wav_path,
            "bytes": len(wav_bytes),
            "content_type": "audio/wav",
            "sample_rate_hz": 16000,
            "channels": 1,
            "sample_width_bytes": 2,
        },
        "analysis_summary": analysis.get("score_summary", {}),
        "targets": targets,
        "pipeline": {
            "asr_provider": "deepgram",
            "asr_model": "nova-2",
            "aligner": "pocketsphinx",
        },
    }

    with open(tmp_json_path, "w", encoding="utf-8") as jf:
        json.dump(sidecar, jf, ensure_ascii=False, indent=2)
        jf.write("\n")
        jf.flush()
        os.fsync(jf.fileno())
    os.replace(tmp_json_path, json_path)

    return {
        "recording_id": recording_id,
        "wav_path": wav_path,
        "json_path": json_path,
    }


def parse_paragraphs(text: str) -> list[dict[str, Any]]:
    """Parse paragraphs and extract target annotations like word[N]/word[V]."""
    # Paragraphs are separated by one or more blank lines.
    parts = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    paragraphs = []
    for i, p in enumerate(parts, start=1):
        words = p.split()
        display_tokens = []
        targets = []
        for idx, w in enumerate(words):
            m = re.match(r"^(.+?)\[([NV])\]([\.,;:!?]?)$", w)
            if m:
                # Keep punctuation for display, but strip the label marker.
                base = m.group(1)
                label = m.group(2)
                punct = m.group(3)
                display = f"{base}{punct}"
                display_tokens.append(display)
                targets.append({
                    "token_index": idx,
                    "word": base,
                    "label": label,
                    "expected_stress": 1 if label == "N" else 2,
                })
            else:
                display_tokens.append(w)
        paragraphs.append({
            "id": i,
            "display_text": " ".join(display_tokens),
            "tokens": display_tokens,
            "targets": targets,
        })
    return paragraphs


with open("5-paragraph-syllable-stress-test_NV.txt", "r", encoding="utf-8") as f:
    PARAGRAPHS = parse_paragraphs(f.read())


def load_pronunciations() -> dict[str, list[list[str]]]:
    """Load CMUdict pronunciations for only target words in test paragraphs."""
    pronunciations: dict[str, list[list[str]]] = {}
    entries = cmudict.dict()
    seen = set()
    for p in PARAGRAPHS:
        for t in p["targets"]:
            w = normalize_token(t["word"])
            if not w or w in seen:
                continue
            seen.add(w)
            variants = []
            for phones in entries.get(w, []):
                # Drop lexical stress digits for phone matching logic.
                ph = [re.sub(r"\d", "", phone) for phone in phones]
                # Keep only multi-syllable candidates (need at least two vowels).
                if sum(1 for x in ph if x in VOWEL_BASES) >= 2:
                    variants.append(ph)
            if variants:
                pronunciations[w] = variants
    return pronunciations


PRONUNCIATIONS = load_pronunciations()


@dataclass
class WavData:
    """Simple container for decoded WAV metadata and PCM payload."""
    sample_rate: int
    channels: int
    sample_width: int
    pcm_bytes: bytes


def read_wav_bytes(wav_bytes: bytes) -> WavData:
    """Decode WAV bytes and return audio properties plus raw PCM frames."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sw = wf.getsampwidth()
        pcm = wf.readframes(wf.getnframes())
    return WavData(sample_rate=sr, channels=channels, sample_width=sw, pcm_bytes=pcm)


def deepgram_transcribe(wav_bytes: bytes) -> list[dict[str, Any]]:
    """Submit WAV to Deepgram and return flattened per-word metadata."""
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav",
    }
    resp = requests.post(DEEPGRAM_URL, headers=headers, data=wav_bytes, timeout=40)
    resp.raise_for_status()
    payload = resp.json()
    words = payload["results"]["channels"][0]["alternatives"][0].get("words", [])
    flat = []
    for idx, word in enumerate(words):
        conf = float(word.get("confidence", 0.0))
        flat.append(
            {
                "idx": idx,
                "word": word.get("word", ""),
                "start": float(word.get("start", 0.0)),
                "end": float(word.get("end", 0.0)),
                "confidence": conf,
                "confidence_cubed": confidence_cubed(conf),
            }
        )
    return flat


def needleman_wunsch_alignment(ref_tokens: list[str], hyp_tokens: list[str]) -> dict[int, int]:
    """Globally align reference/hypothesis tokens and return exact-match mapping."""
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]

    gap = -1
    for i in range(1, n + 1):
        dp[i][0] = i * gap
        bt[i][0] = "U"
    for j in range(1, m + 1):
        dp[0][j] = j * gap
        bt[0][j] = "L"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_score = 2 if ref_tokens[i - 1] == hyp_tokens[j - 1] else -1
            choices = [
                (dp[i - 1][j - 1] + match_score, "D"),
                (dp[i - 1][j] + gap, "U"),
                (dp[i][j - 1] + gap, "L"),
            ]
            best = max(choices, key=lambda x: x[0])
            dp[i][j], bt[i][j] = best

    mapping: dict[int, int] = {}
    i, j = n, m
    while i > 0 or j > 0:
        step = bt[i][j]
        if step == "D":
            # Only map aligned positions when the normalized tokens truly match.
            if i > 0 and j > 0 and ref_tokens[i - 1] == hyp_tokens[j - 1]:
                mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif step == "U":
            i -= 1
        else:
            j -= 1
    return mapping


def _ensure_decoder():
    """Lazily initialize and cache the PocketSphinx decoder instance."""
    global _DECODER
    if _DECODER is not None:
        return _DECODER
    from pocketsphinx import Decoder, get_model_path

    model_path = get_model_path()
    _DECODER = Decoder(
        bestpath=False,
        hmm=model_path + "/en-us/en-us",
        dict=model_path + "/en-us/cmudict-en-us.dict",
        lm=None,
        loglevel="FATAL",
        fsgusefiller=False,
    )
    return _DECODER


def slice_word_pcm(wav_data: WavData, start: float, end: float, total_duration: float) -> bytes:
    """Extract a padded PCM slice for a word-level audio segment."""
    if wav_data.sample_rate != 16000 or wav_data.channels != 1 or wav_data.sample_width != 2:
        raise ValueError("Input audio must be 16kHz mono 16-bit PCM WAV")

    pad_start = max(0.0, start - MAX_WORD_PAD_SEC)
    pad_end = min(total_duration, end + MAX_WORD_PAD_SEC)
    bps = wav_data.sample_width * wav_data.channels
    s_idx = int(pad_start * wav_data.sample_rate)
    e_idx = int(pad_end * wav_data.sample_rate)
    return wav_data.pcm_bytes[s_idx * bps : e_idx * bps]


def align_phonemes(word_text: str, pcm_bytes: bytes, start_time: float) -> list[dict[str, Any]]:
    """Run forced alignment for one word and return phone intervals in seconds."""
    decoder = _ensure_decoder()
    decoder.set_align_text(normalize_token(word_text) or word_text)
    decoder.start_utt()
    decoder.process_raw(pcm_bytes, full_utt=True)
    decoder.end_utt()

    decoder.set_alignment()
    decoder.start_utt()
    decoder.process_raw(pcm_bytes, full_utt=True)
    decoder.end_utt()

    spx_words = decoder.get_alignment()
    phones: list[dict[str, Any]] = []
    for sword in spx_words or []:
        for phone in sword:
            if phone.name == "SIL":
                continue
            p_start = start_time + (phone.start * 0.01) - MAX_WORD_PAD_SEC
            p_end = start_time + ((phone.start + phone.duration) * 0.01) - MAX_WORD_PAD_SEC
            phones.append({"phone": phone.name, "start": max(0.0, p_start), "end": max(0.0, p_end)})
    return phones


def phoneme_match_score(aligned: list[dict[str, Any]], pronunciation: list[str]) -> int:
    """Score phone sequence overlap by position-wise matches."""
    aligned_seq = [re.sub(r"\d", "", p["phone"]) for p in aligned]
    pron_seq = [re.sub(r"\d", "", x) for x in pronunciation]
    return sum(1 for i in range(min(len(aligned_seq), len(pron_seq))) if aligned_seq[i] == pron_seq[i])


def infer_stress_from_word(word: str, aligned_phones: list[dict[str, Any]]) -> dict[str, Any]:
    """Infer stress from aligned vowels by comparing first two vowel durations."""
    norm = normalize_token(word)
    variants = PRONUNCIATIONS.get(norm, [])
    if not variants or not aligned_phones:
        return {
            "inferred_stress": None,
            "core_durations": {"syll1": None, "syll2": None},
            "core_phones": {"syll1": None, "syll2": None},
        }

    best = max(variants, key=lambda v: phoneme_match_score(aligned_phones, v))
    vowel_positions = [(i, p) for i, p in enumerate(best) if re.sub(r"\d", "", p) in VOWEL_BASES]
    if len(vowel_positions) < 2:
        return {
            "inferred_stress": None,
            "core_durations": {"syll1": None, "syll2": None},
            "core_phones": {"syll1": None, "syll2": None},
        }

    syll1_phone = vowel_positions[0][1]
    syll2_phone = vowel_positions[1][1]
    pos1 = vowel_positions[0][0]
    pos2 = vowel_positions[1][0]

    d1 = None
    d2 = None
    if pos1 < len(aligned_phones):
        d1 = max(0.0, aligned_phones[pos1]["end"] - aligned_phones[pos1]["start"])
    if pos2 < len(aligned_phones):
        d2 = max(0.0, aligned_phones[pos2]["end"] - aligned_phones[pos2]["start"])

    if d1 is not None:
        d1 = round(d1, 2)
    if d2 is not None:
        d2 = round(d2, 2)

    inferred = None
    if d1 is not None and d2 is not None:
        # Simple duration heuristic: longer nucleus is treated as stressed.
        inferred = 1 if d1 >= d2 else 2

    return {
        "inferred_stress": inferred,
        "core_durations": {"syll1": d1, "syll2": d2},
        "core_phones": {"syll1": syll1_phone, "syll2": syll2_phone},
    }


def build_render_words(display_text: str, alignment: dict[int, int], deepgram_words: list[dict[str, Any]], targets: list[dict[str, Any]], target_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create UI-ready token metadata while preserving original spacing."""
    target_by_idx = {t["token_index"]: t for t in targets}
    result_by_idx = {t["token_index"]: r for t, r in zip(targets, target_results)}
    render = []
    word_i = 0
    for chunk in re.findall(r"\S+|\s+", display_text):
        if chunk.isspace():
            render.append({"text": chunk, "is_space": True})
            continue
        dg_idx = alignment.get(word_i)
        dg = deepgram_words[dg_idx] if dg_idx is not None and dg_idx < len(deepgram_words) else None
        tr = result_by_idx.get(word_i)
        render.append({
            "text": chunk,
            "is_space": False,
            "deepgram_idx": dg_idx,
            "confidence_cubed": dg["confidence_cubed"] if dg else None,
            "bg_norm": normalize_bg(dg["confidence_cubed"] if dg else None),
            "is_target": word_i in target_by_idx,
            "target_status": tr["status"] if tr else None,
            "target_correct": tr["correct"] if tr else None,
        })
        word_i += 1
    return render


def analyze_payload(paragraph: dict[str, Any], wav_bytes: bytes) -> dict[str, Any]:
    """End-to-end analysis pipeline for a paragraph and uploaded WAV audio."""
    deepgram_words = deepgram_transcribe(wav_bytes)
    ref_norm = [normalize_token(t) for t in paragraph["tokens"]]
    hyp_norm = [normalize_token(w["word"]) for w in deepgram_words]
    alignment = needleman_wunsch_alignment(ref_norm, hyp_norm)

    wav_data = read_wav_bytes(wav_bytes)
    duration = len(wav_data.pcm_bytes) / (wav_data.sample_rate * wav_data.channels * wav_data.sample_width)

    targets_out = []
    for t in paragraph["targets"]:
        # Start each target with a pessimistic default that is refined if aligned.
        mapped = alignment.get(t["token_index"])
        base = {
            "token_index": t["token_index"],
            "word": t["word"],
            "label": t["label"],
            "expected_stress": t["expected_stress"],
            "inferred_stress": None,
            "correct": None,
            "status": "missing",
            "core_durations": {"syll1": None, "syll2": None},
            "core_phones": {"syll1": None, "syll2": None},
            "deepgram_word_index": None,
            "deepgram_confidence": None,
            "deepgram_confidence_cubed": None,
            "feedback": "Word not matched in transcript. Re-read clearly with target stress.",
        }
        if mapped is None or mapped >= len(deepgram_words):
            targets_out.append(base)
            continue

        dg = deepgram_words[mapped]
        base["deepgram_word_index"] = mapped
        base["deepgram_confidence"] = dg["confidence"]
        base["deepgram_confidence_cubed"] = dg["confidence_cubed"]
        base["status"] = "unaligned"

        try:
            segment = slice_word_pcm(wav_data, dg["start"], dg["end"], duration)
            phones = align_phonemes(t["word"], segment, dg["start"])
            inferred = infer_stress_from_word(t["word"], phones)
            base.update(inferred)
            if inferred["inferred_stress"] is not None:
                base["status"] = "ok"
                base["correct"] = inferred["inferred_stress"] == t["expected_stress"]
                base["feedback"] = (
                    "Good stress placement."
                    if base["correct"]
                    else "Shift stress to syllable {} and lengthen that vowel.".format(t["expected_stress"])
                )
            else:
                base["feedback"] = "Could not infer stress; speak the target word more clearly."
        except Exception as exc:
            # Keep analysis resilient: one target failure should not abort the batch.
            log(f"alignment failed for word={t['word']}: {exc}")
            base["feedback"] = "Alignment failed for this word segment."

        targets_out.append(base)

    scored = [t for t in targets_out if t["status"] == "ok" and t["correct"] is not None]
    missing = [t for t in targets_out if t["status"] == "missing"]
    unaligned = [t for t in targets_out if t["status"] == "unaligned"]
    correct_count = sum(1 for t in scored if t["correct"])
    pct = (100.0 * correct_count / len(scored)) if scored else 0.0

    render_words = build_render_words(paragraph["display_text"], alignment, deepgram_words, paragraph["targets"], targets_out)

    return {
        "deepgram_words": deepgram_words,
        "alignment": [{"paragraph_token_index": k, "deepgram_word_index": v} for k, v in sorted(alignment.items())],
        "targets": targets_out,
        "score_summary": {
            "percent_correct": round(pct, 2),
            "total_targets": len(paragraph["targets"]),
            "scored_targets": len(scored),
            "missing_targets": len(missing),
            "unaligned_targets": len(unaligned),
        },
        "render_words": render_words,
    }




def get_client_ip() -> str | None:
    """Resolve client IP from Cloud Run proxy headers when available."""
    forwarded_for = request.headers.get("X-Forwarded-For", "").strip()
    if forwarded_for:
        return forwarded_for.split(",")[0].strip() or None
    return request.remote_addr

def get_request_id() -> str:
    """Use inbound request id when available; otherwise generate one."""
    header = request.headers.get("X-Request-Id", "").strip()
    return header or str(uuid.uuid4())


@app.before_request
def attach_request_id() -> None:
    """Attach request context fields to flask.g before request handling."""
    g.request_id = get_request_id()
    g.client_ip = get_client_ip()


@app.after_request
def add_request_id_header(resp: Response) -> Response:
    """Mirror server request id in response headers for client correlation."""
    resp.headers["X-Request-Id"] = g.request_id
    return resp


@app.route("/")
def index() -> str:
    """Serve the web UI."""
    return render_template("index.html")


@app.route("/api/paragraphs")
def api_paragraphs():
    """Return all parsed practice paragraphs with target metadata."""
    return jsonify({"request_id": g.request_id, "paragraphs": PARAGRAPHS})


@app.route("/healthz")
def healthz():
    """Lightweight health-check endpoint."""
    return jsonify({"request_id": g.request_id, "status": "ok"})


def _agent_card() -> dict[str, Any]:
    """Build static-ish A2A agent capability metadata."""
    base_url = request.host_url.rstrip("/")
    return {
        "name": "Syllable Stress Evaluator",
        "description": "Evaluates noun/verb stress-shift pronunciation from WAV audio and paragraph context.",
        "version": "0.1.0",
        "protocolVersion": "0.1",
        "url": base_url,
        "skills": [
            {
                "name": "pronunciation.evaluate",
                "description": "Evaluate syllable stress for annotated noun/verb targets using 16kHz mono WAV.",
            }
        ],
        "defaultInputModes": ["audio/wav", "application/json"],
        "defaultOutputModes": ["application/json"],
    }


@app.route("/.well-known/agent-card.json")
def agent_card():
    """Expose A2A agent-card metadata at the well-known endpoint."""
    card = _agent_card()
    card["request_id"] = g.request_id
    return jsonify(card)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Handle multipart form upload and return full stress analysis JSON."""
    paragraph_raw = request.form.get("paragraph_id", "0")
    try:
        paragraph_id = int(paragraph_raw)
    except (TypeError, ValueError):
        return jsonify({"request_id": g.request_id, "error": "paragraph_id must be 1..5"}), 400
    audio = request.files.get("audio_wav")
    if paragraph_id < 1 or paragraph_id > len(PARAGRAPHS):
        return jsonify({"request_id": g.request_id, "error": "paragraph_id must be 1..5"}), 400
    if not audio:
        return jsonify({"request_id": g.request_id, "error": "audio_wav is required"}), 400
    native_exemplar = parse_bool(request.form.get("native_exemplar"))

    wav_bytes = audio.read()
    try:
        analysis = analyze_payload(PARAGRAPHS[paragraph_id - 1], wav_bytes)
        persisted = persist_submission(
            wav_bytes=wav_bytes,
            paragraph=PARAGRAPHS[paragraph_id - 1],
            paragraph_id=paragraph_id,
            analysis=analysis,
            native_exemplar=native_exemplar,
            source="web",
        )
        analysis["request_id"] = g.request_id
        analysis["persistence"] = persisted
        log(f"analyze completed paragraph_id={paragraph_id} words={len(analysis['deepgram_words'])}")
        return jsonify(analysis)
    except Exception as exc:
        log(f"analyze failed: {exc}")
        return jsonify({"request_id": g.request_id, "error": str(exc)}), 500


@app.route("/a2a", methods=["POST"])
def a2a():
    """JSON-RPC style endpoint for remote pronunciation evaluation."""
    rpc = request.get_json(silent=True) or {}
    rpc_id = rpc.get("id")
    method = rpc.get("method")
    params = rpc.get("params") or {}

    def rpc_err(code: int, message: str, status: int = 400):
        """Return a JSON-RPC error object with request context."""
        return jsonify({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message, "request_id": g.request_id}}), status

    if rpc.get("jsonrpc") != "2.0":
        return rpc_err(-32600, "Invalid JSON-RPC version")

    if method == "agent.about":
        result = _agent_card()
        result["request_id"] = g.request_id
        return jsonify({"jsonrpc": "2.0", "id": rpc_id, "result": result})

    if method == "pronunciation.evaluate":
        p_id = params.get("paragraph_id")
        p_text = params.get("paragraph_text")
        b64 = params.get("audio_wav_base64")
        native_exemplar = parse_bool(params.get("native_exemplar"))
        if not b64:
            return rpc_err(-32602, "audio_wav_base64 is required")
        if not p_id and not p_text:
            return rpc_err(-32602, "paragraph_id or paragraph_text is required")

        try:
            wav_bytes = base64.b64decode(b64)
        except Exception:
            return rpc_err(-32602, "audio_wav_base64 must be valid base64")

        if p_text:
            paragraph = parse_paragraphs(p_text)[0]
        else:
            if not isinstance(p_id, int) or p_id < 1 or p_id > len(PARAGRAPHS):
                return rpc_err(-32602, "paragraph_id must be an int in range 1..5")
            paragraph = PARAGRAPHS[p_id - 1]

        try:
            analysis = analyze_payload(paragraph, wav_bytes)
            analysis.pop("render_words", None)
            persisted = persist_submission(
                wav_bytes=wav_bytes,
                paragraph=paragraph,
                paragraph_id=p_id if isinstance(p_id, int) else None,
                analysis=analysis,
                native_exemplar=native_exemplar,
                source="a2a",
            )
            return jsonify({
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {"request_id": g.request_id, "analysis": analysis, "persistence": persisted},
            })
        except Exception as exc:
            log(f"a2a eval failed: {exc}")
            return rpc_err(-32000, str(exc), status=500)

    return rpc_err(-32601, "Method not found", status=404)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
