"""Flask application for evaluating noun/verb syllable stress from WAV audio.

The service accepts paragraph context with marked target words and aligns ASR
output and phone-level timing to infer whether stress was placed on syllable 1
or syllable 2 for each target.
"""

import base64, hashlib, io, json, math, os, re, statistics, sys, time, uuid, wave
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any

import cmudict
import requests
from flask import Flask, Response, g, has_request_context, jsonify, render_template, request


# Shared fallback Deepgram API key used for outbound ASR requests.
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
# Deepgram endpoint with explicit query params.
# Keeping model/language/punctuation flags fixed reduces variation between runs,
# which makes stress-threshold tuning and regression debugging less noisy.
DEEPGRAM_URL = (
    "https://api.deepgram.com/v1/listen"
    "?model=nova-2&language=en-US&punctuate=false&diarize=false"
)
# ARPAbet vowel phone set used to identify candidate stressed nuclei.
VOWEL_BASES = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"
}
# Padding around each target-word window during phone alignment.
MAX_WORD_PAD_SEC = 0.12
# Bucket directory where WAV/JSON sidecar pairs are persisted and reloaded from.
BUCKET_DIR = os.environ.get("BUCKET_DIR", "/bucket")
# Timezone used for recording ids and persisted timestamps.
HST_TZ = ZoneInfo("Pacific/Honolulu")
# Small epsilon to stabilize log-ratio computation when durations are very small.
RATIO_EPS = 1e-4
# Minimum native-exemplar samples per class (noun/verb) before using learned thresholds.
MIN_EXEMPLARS_PER_CLASS = 2
# In-process cache TTL for learned thresholds.
# Sidecar scans can dominate latency, so we trade freshness for predictable read cost.
THRESHOLD_CACHE_TTL_SEC = 60


# Flask application instance for all HTTP and A2A routes.
app = Flask(__name__)
# Maximum request payload size (25 MB) to guard upload resource usage.
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


# Lazy-initialized PocketSphinx decoder singleton.
_DECODER = None
# Learned-threshold cache keyed by context/word, refreshed on TTL expiry.
_THRESHOLD_CACHE: dict[str, Any] = {"loaded_at": 0.0, "bucket_dir": None, "thresholds": {}}


class DeepgramAPIError(RuntimeError):
    """Friendly, client-safe wrapper for upstream Deepgram API failures."""

    def __init__(self, user_message: str, status_code: int | None = None):
        super().__init__(user_message)
        self.user_message = user_message
        self.status_code = status_code


def _track_timing(metric: str, elapsed_sec: float) -> None:
    """Accumulate request-scoped timing metrics when a request context is active."""
    if not has_request_context():
        return
    timings = getattr(g, "timings", None)
    if not isinstance(timings, dict):
        return
    timings[metric] = float(timings.get(metric, 0.0)) + float(elapsed_sec)


def log(message: str) -> None:
    """Print request-scoped logs with a request id for easier traceability."""
    rid = getattr(g, "request_id", "-")
    print(f"[request_id={rid}] {message}", file=sys.stderr, flush=True)


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


def _threshold_key(word_norm: str, paragraph_id: int | None = None, token_index: int | None = None) -> str:
    """Build a stable key string for learned-threshold lookup."""
    if paragraph_id is None or token_index is None:
        return f"word:{word_norm}"
    return f"ctx:{word_norm}:{paragraph_id}:{token_index}"


def _load_adaptive_thresholds(bucket_dir: str) -> dict[str, dict[str, Any]]:
    """Scan sidecars in bucket and compute median-midpoint stress thresholds."""
    # Local import keeps module import light and avoids global dependency for tests.
    import glob

    # grouped[key][class] -> list[duration_ratio_log].
    # class is expected stress (1=noun pattern, 2=verb pattern).
    grouped: dict[str, dict[int, list[float]]] = {}
    # Iterate every persisted sidecar in the mounted bucket.
    for json_path in glob.glob(os.path.join(bucket_dir, "*.json")):
        try:
            # Skip malformed/partial sidecars and continue.
            # This keeps one bad write from disabling adaptive thresholds globally.
            with open(json_path, "r", encoding="utf-8") as fh:
                sidecar = json.load(fh)
        except Exception:
            continue

        if not isinstance(sidecar, dict) or sidecar.get("native_exemplar") is not True:
            continue

        # paragraph_id is optional (e.g., free-text A2A calls), so keep nullable.
        paragraph_id = sidecar.get("paragraph_id") if isinstance(sidecar.get("paragraph_id"), int) else None
        # Evaluate each target row as a potential training sample.
        for target in sidecar.get("targets", []):
            if not isinstance(target, dict):
                continue
            if target.get("status") != "ok":
                continue
            expected_stress = target.get("expected_stress")
            if expected_stress not in (1, 2):
                continue
            word_norm = normalize_token(target.get("word_norm") or target.get("word_display") or "")
            if not word_norm:
                continue

            d1 = target.get("core_durations", {}).get("syll1") if isinstance(target.get("core_durations"), dict) else None
            d2 = target.get("core_durations", {}).get("syll2") if isinstance(target.get("core_durations"), dict) else None
            ratio_log = target.get("duration_ratio_log")
            if ratio_log is None:
                _, ratio_log = _ratio_metrics(d1, d2)
            if ratio_log is None:
                continue

            # Always collect a word-level key so we can generalize across contexts.
            word_key = _threshold_key(word_norm)
            grouped.setdefault(word_key, {1: [], 2: []})[expected_stress].append(float(ratio_log))

            # Also collect context-level key when paragraph/token coordinates are available.
            token_index = target.get("token_index") if isinstance(target.get("token_index"), int) else None
            if paragraph_id is not None and token_index is not None:
                ctx_key = _threshold_key(word_norm, paragraph_id, token_index)
                grouped.setdefault(ctx_key, {1: [], 2: []})[expected_stress].append(float(ratio_log))

    # Final learned thresholds keyed by context (preferred) and word (fallback).
    # Threshold is midpoint of class medians in log-ratio space.
    learned: dict[str, dict[str, Any]] = {}
    for key, classes in grouped.items():
        c1 = classes[1]
        c2 = classes[2]
        if len(c1) < MIN_EXEMPLARS_PER_CLASS or len(c2) < MIN_EXEMPLARS_PER_CLASS:
            continue
        threshold = (statistics.median(c1) + statistics.median(c2)) / 2.0
        learned[key] = {
            "threshold": round(float(threshold), 6),
            "class1_count": len(c1),
            "class2_count": len(c2),
            "total": len(c1) + len(c2),
            "key": key,
        }
    return learned


def get_learned_threshold(word_norm: str, paragraph_id: int | None = None, token_index: int | None = None) -> dict[str, Any] | None:
    """Get cached learned threshold stats for a word/context when available."""
    # Monotonic clock avoids issues with wall-clock adjustments.
    now = time.monotonic()
    should_reload = (
        _THRESHOLD_CACHE.get("bucket_dir") != BUCKET_DIR
        or (now - float(_THRESHOLD_CACHE.get("loaded_at", 0.0))) >= THRESHOLD_CACHE_TTL_SEC
    )
    if should_reload:
        load_start = time.perf_counter()
        try:
            # Refresh cache from the bucket directory on TTL expiry/bucket switch.
            _THRESHOLD_CACHE["thresholds"] = _load_adaptive_thresholds(BUCKET_DIR)
            _THRESHOLD_CACHE["loaded_at"] = now
            _THRESHOLD_CACHE["bucket_dir"] = BUCKET_DIR
        except Exception as exc:
            # Fail open: preserve request path even if threshold loading fails.
            # Downstream code falls back to naive duration comparison.
            log(f"adaptive threshold load warning: {exc}")
            _THRESHOLD_CACHE["thresholds"] = {}
            _THRESHOLD_CACHE["loaded_at"] = now
            _THRESHOLD_CACHE["bucket_dir"] = BUCKET_DIR
        finally:
            _track_timing("bucket_json_read_process_sec", time.perf_counter() - load_start)

    thresholds = _THRESHOLD_CACHE.get("thresholds", {})
    ctx_key = _threshold_key(word_norm, paragraph_id, token_index)
    return thresholds.get(ctx_key) or thresholds.get(_threshold_key(word_norm))


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

    write_start = time.perf_counter()
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
    _track_timing("persist_output_files_sec", time.perf_counter() - write_start)

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


# Load canonical practice paragraphs once at import-time for consistent ids/targets.
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


# Build a compact pronunciation inventory only for target words in configured paragraphs.
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


def resolve_deepgram_api_key(explicit_key: str | None = None) -> str:
    """Resolve Deepgram key source with precedence: explicit, cookie, env."""
    if explicit_key and explicit_key.strip():
        log("Using Deepgram key from: a2a_param")
        return explicit_key.strip()

    # Intentionally read a JS-managed cookie (HttpOnly=False) for demo/dev UX.
    # This is acceptable for this tool, but production deployments should
    # reconsider cookie scope/storage strategy and threat model.
    cookie_key = request.cookies.get("deepgram_api_key", "") if has_request_context() else ""
    if cookie_key and cookie_key.strip():
        log("Using Deepgram key from: cookie")
        return cookie_key.strip()

    if DEEPGRAM_API_KEY and DEEPGRAM_API_KEY.strip():
        log("Using Deepgram key from: env")
        return DEEPGRAM_API_KEY.strip()

    raise RuntimeError("No Deepgram API key provided (cookie, A2A param, or DEEPGRAM_API_KEY env)")


def deepgram_transcribe(wav_bytes: bytes, deepgram_api_key: str | None = None) -> list[dict[str, Any]]:
    """Submit WAV to Deepgram and return flattened per-word metadata."""
    resolved_key = resolve_deepgram_api_key(deepgram_api_key)
    headers = {
        "Authorization": f"Token {resolved_key}",
        "Content-Type": "audio/wav",
    }
    try:
        resp = requests.post(DEEPGRAM_URL, headers=headers, data=wav_bytes, timeout=40)
        resp.raise_for_status()
    except requests.HTTPError:
        status = resp.status_code if "resp" in locals() else None
        if status == 400:
            raise DeepgramAPIError(
                "Deepgram could not process this audio request (400 Bad Request). Please verify the media format and try again.",
                status_code=400,
            ) from None
        if status == 401:
            raise DeepgramAPIError(
                "Deepgram rejected this API key (401 Unauthorized). Please set a working API key and submit again.",
                status_code=401,
            ) from None
        if status == 402:
            raise DeepgramAPIError(
                "Deepgram reports this API key is out of funds (402 Payment Required). Please top up credits or use a different key.",
                status_code=402,
            ) from None
        if status == 429:
            raise DeepgramAPIError(
                "Deepgram rate limit reached (429 Too Many Requests). Please wait briefly, then press Submit for Analysis again.",
                status_code=429,
            ) from None
        raise DeepgramAPIError(
            f"Deepgram request failed (HTTP {status or 'unknown'}). Please verify your API key and try again.",
            status_code=status,
        ) from None
    except requests.RequestException:
        raise DeepgramAPIError(
            "Deepgram request failed due to a network or service issue. Please try again.",
            status_code=None,
        ) from None
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


def infer_stress_from_word(
    word: str,
    aligned_phones: list[dict[str, Any]],
    paragraph_id: int | None = None,
    token_index: int | None = None,
) -> dict[str, Any]:
    """Infer stress using learned thresholds when available, otherwise naive duration."""
    norm = normalize_token(word)
    variants = PRONUNCIATIONS.get(norm, [])
    if not variants or not aligned_phones:
        return {
            "inferred_stress": None,
            "core_durations": {"syll1": None, "syll2": None},
            "core_phones": {"syll1": None, "syll2": None},
            "duration_ratio_log": None,
            "learned_threshold": None,
            "threshold_key": None,
            "decision_method": None,
        }

    best = max(variants, key=lambda v: phoneme_match_score(aligned_phones, v))
    vowel_positions = [(i, p) for i, p in enumerate(best) if re.sub(r"\d", "", p) in VOWEL_BASES]
    if len(vowel_positions) < 2:
        return {
            "inferred_stress": None,
            "core_durations": {"syll1": None, "syll2": None},
            "core_phones": {"syll1": None, "syll2": None},
            "duration_ratio_log": None,
            "learned_threshold": None,
            "threshold_key": None,
            "decision_method": None,
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
    ratio_log = None
    learned_threshold = None
    threshold_key = None
    decision_method = None
    if d1 is not None and d2 is not None:
        _, ratio_log = _ratio_metrics(d1, d2)
        threshold_stats = get_learned_threshold(norm, paragraph_id=paragraph_id, token_index=token_index)
        if threshold_stats and ratio_log is not None:
            learned_threshold = threshold_stats["threshold"]
            threshold_key = threshold_stats["key"]
            inferred = 1 if ratio_log >= learned_threshold else 2
            decision_method = "learned_threshold"
        else:
            inferred = 1 if d1 >= d2 else 2
            decision_method = "naive_duration"

    return {
        "inferred_stress": inferred,
        "core_durations": {"syll1": d1, "syll2": d2},
        "core_phones": {"syll1": syll1_phone, "syll2": syll2_phone},
        "duration_ratio_log": ratio_log,
        "learned_threshold": learned_threshold,
        "threshold_key": threshold_key,
        "decision_method": decision_method,
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


def analyze_payload(paragraph: dict[str, Any], wav_bytes: bytes, deepgram_api_key: str | None = None) -> dict[str, Any]:
    """End-to-end analysis pipeline for a paragraph and uploaded WAV audio."""
    # 1) ASR transcript with timestamps/confidence per recognized word.
    deepgram_start = time.perf_counter()
    deepgram_words = deepgram_transcribe(wav_bytes, deepgram_api_key=deepgram_api_key)
    deepgram_elapsed = time.perf_counter() - deepgram_start
    # 2) Normalize tokens and align paragraph words to ASR words.
    ref_norm = [normalize_token(t) for t in paragraph["tokens"]]
    hyp_norm = [normalize_token(w["word"]) for w in deepgram_words]
    alignment = needleman_wunsch_alignment(ref_norm, hyp_norm)

    # 3) Decode WAV once so target-level slicing can reuse PCM bytes.
    wav_data = read_wav_bytes(wav_bytes)
    duration = len(wav_data.pcm_bytes) / (wav_data.sample_rate * wav_data.channels * wav_data.sample_width)

    alignment_total = 0.0
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
            "decision_method": None,
            "duration_ratio_log": None,
            "learned_threshold": None,
            "threshold_key": None,
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
            alignment_start = time.perf_counter()
            phones = align_phonemes(t["word"], segment, dg["start"])
            alignment_total += time.perf_counter() - alignment_start
            inferred = infer_stress_from_word(t["word"], phones, paragraph_id=paragraph.get("id"), token_index=t.get("token_index"))
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

    # 4) Aggregate scoring metrics for summary and UI.
    scored = [t for t in targets_out if t["status"] == "ok" and t["correct"] is not None]
    missing = [t for t in targets_out if t["status"] == "missing"]
    unaligned = [t for t in targets_out if t["status"] == "unaligned"]
    correct_count = sum(1 for t in scored if t["correct"])
    pct = (100.0 * correct_count / len(scored)) if scored else 0.0

    render_words = build_render_words(paragraph["display_text"], alignment, deepgram_words, paragraph["targets"], targets_out)

    timing = {
        "recording_duration_sec": round(duration, 3),
        "bucket_json_read_process_sec": round(float(getattr(g, "timings", {}).get("bucket_json_read_process_sec", 0.0)), 3),
        "deepgram_api_sec": round(deepgram_elapsed, 3),
        "pocketsphinx_alignment_sec": round(alignment_total, 3),
        "persist_output_files_sec": round(float(getattr(g, "timings", {}).get("persist_output_files_sec", 0.0)), 3),
    }

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
        "timing": timing,
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
    g.timings = {}


@app.after_request
def add_request_id_header(resp: Response) -> Response:
    """Mirror server request id in response headers for client correlation."""
    resp.headers["X-Request-Id"] = g.request_id
    return resp


@app.route("/")
def index() -> str:
    """Serve the web UI."""
    return render_template("index.html", base_url=request.host_url.rstrip("/"))


@app.route("/api/paragraphs")
def api_paragraphs():
    """Return all parsed practice paragraphs with target metadata."""
    return jsonify({"request_id": g.request_id, "paragraphs": PARAGRAPHS})


@app.route("/healthz")
def healthz():
    """Lightweight health-check endpoint."""
    return jsonify({"request_id": g.request_id, "status": "ok"})


@app.route("/robots.txt")
def robots_txt() -> Response:
    """Serve a fully permissive robots policy for all crawlers."""
    body = "User-agent: *\nAllow: /\n"
    return Response(body, mimetype="text/plain")


def _agent_card() -> dict[str, Any]:
    """Build static-ish A2A agent capability metadata."""
    base_url = request.host_url.rstrip("/")
    rpc_endpoint = f"{base_url}/a2a"
    return {
        "name": "Syllable Stress Evaluator",
        "description": "Evaluates noun/verb stress-shift pronunciation from WAV audio and paragraph context.",
        "version": "0.4.0",
        "protocolVersion": "0.1",
        "url": base_url,
        "documentationUrl": f"{base_url}/",
        "capabilities": {
            "jsonrpcEndpoint": rpc_endpoint,
            "methods": {
                "agent.about": {
                    "description": "Return model card metadata and runtime request_id.",
                },
                "pronunciation.evaluate": {
                    "description": "Evaluate a 16kHz mono WAV against paragraph_id or paragraph_text.",
                    "requiredParams": ["audio_wav_base64"],
                    "optionalParams": ["paragraph_id", "paragraph_text", "native_exemplar", "deepgram_api_key"],
                },
                "paragraphs.count": {
                    "description": "Return the number of configured practice paragraphs.",
                },
                "paragraphs.get_text": {
                    "description": "Return plain unannotated paragraph text for a given paragraph_id.",
                    "requiredParams": ["paragraph_id"],
                },
            },
        },
        "skills": [
            {
                "name": "pronunciation.evaluate",
                "description": "Evaluate syllable stress for annotated noun/verb targets using 16kHz mono WAV.",
            }
        ],
        "defaultInputModes": ["audio/wav", "application/json"],
        "defaultOutputModes": ["application/json"],
    }


@app.route("/.well-known/agent.json")
def agent_card():
    """Expose A2A agent-card metadata at the well-known endpoint."""
    card = _agent_card()
    card["request_id"] = g.request_id
    return jsonify(card)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Handle multipart form upload and return full stress analysis JSON."""
    # Parse and validate paragraph id early for fast failure on malformed requests.
    paragraph_raw = request.form.get("paragraph_id", "0")
    paragraph_range_msg = f"paragraph_id must be 1..{len(PARAGRAPHS)}"
    try:
        paragraph_id = int(paragraph_raw)
    except (TypeError, ValueError):
        return jsonify({"request_id": g.request_id, "error": paragraph_range_msg}), 400
    audio = request.files.get("audio_wav")
    if paragraph_id < 1 or paragraph_id > len(PARAGRAPHS):
        return jsonify({"request_id": g.request_id, "error": paragraph_range_msg}), 400
    if not audio:
        return jsonify({"request_id": g.request_id, "error": "audio_wav is required"}), 400
    native_exemplar = parse_bool(request.form.get("native_exemplar"))

    # Read full WAV payload and run analysis + persistence in one protected block.
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
        analysis.setdefault("timing", {})["persist_output_files_sec"] = round(
            float(getattr(g, "timings", {}).get("persist_output_files_sec", 0.0)), 3
        )
        analysis["request_id"] = g.request_id
        analysis["persistence"] = persisted
        log(f"analyze completed paragraph_id={paragraph_id} words={len(analysis['deepgram_words'])}")
        return jsonify(analysis)
    except DeepgramAPIError as exc:
        log(f"analyze failed deepgram status={exc.status_code} message={exc.user_message}")
        http_status = exc.status_code if exc.status_code in (400, 401, 402, 429) else 502
        return jsonify({"request_id": g.request_id, "error": exc.user_message, "error_code": exc.status_code}), http_status
    except Exception as exc:
        log(f"analyze failed: {exc}")
        return jsonify({"request_id": g.request_id, "error": str(exc)}), 500


@app.route("/a2a", methods=["POST"])
def a2a():
    """JSON-RPC style endpoint for remote pronunciation evaluation."""
    # Decode request envelope with permissive defaults to preserve compatibility.
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

    if method == "paragraphs.count":
        return jsonify({
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {"request_id": g.request_id, "paragraph_count": len(PARAGRAPHS)},
        })

    if method == "paragraphs.get_text":
        p_id = params.get("paragraph_id")
        if not isinstance(p_id, int) or p_id < 1 or p_id > len(PARAGRAPHS):
            return rpc_err(-32602, f"paragraph_id must be an int in range 1..{len(PARAGRAPHS)}")
        paragraph = PARAGRAPHS[p_id - 1]
        return jsonify({
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "request_id": g.request_id,
                "paragraph_id": p_id,
                "paragraph_text": paragraph["display_text"],
            },
        })

    if method == "pronunciation.evaluate":
        p_id = params.get("paragraph_id")
        p_text = params.get("paragraph_text")
        b64 = params.get("audio_wav_base64")
        native_exemplar = parse_bool(params.get("native_exemplar"))
        deepgram_api_key = params.get("deepgram_api_key")
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
                return rpc_err(-32602, f"paragraph_id must be an int in range 1..{len(PARAGRAPHS)}")
            paragraph = PARAGRAPHS[p_id - 1]

        try:
            if isinstance(deepgram_api_key, str) and deepgram_api_key.strip():
                analysis = analyze_payload(paragraph, wav_bytes, deepgram_api_key=deepgram_api_key)
            else:
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
            analysis.setdefault("timing", {})["persist_output_files_sec"] = round(
                float(getattr(g, "timings", {}).get("persist_output_files_sec", 0.0)), 3
            )
            return jsonify({
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {"request_id": g.request_id, "analysis": analysis, "persistence": persisted},
            })
        except DeepgramAPIError as exc:
            log(f"a2a eval failed deepgram status={exc.status_code} message={exc.user_message}")
            if exc.status_code == 400:
                return rpc_err(-32602, exc.user_message, status=400)
            if exc.status_code in (401, 402, 429):
                return rpc_err(-32000, exc.user_message, status=exc.status_code)
            return rpc_err(-32000, exc.user_message, status=502)
        except Exception as exc:
            log(f"a2a eval failed: {exc}")
            return rpc_err(-32000, str(exc), status=500)

    return rpc_err(-32601, "Method not found", status=404)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
