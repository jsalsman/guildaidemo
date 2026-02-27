# syllable-stress

Single-page EFL pronunciation evaluator for noun/verb stress-shift pairs, with A2A-compatible discovery + JSON-RPC endpoints.

Try it: https://guildaidemo.talknicer.com

## Features

- Paragraph selector for 5 annotated paragraphs from `5-paragraph-syllable-stress-test_NV.txt`.
- Browser microphone recording with WAV encoding to **16kHz / 16-bit PCM / mono**.
- "native exemplar" checkbox in the UI to mark exemplar-candidate submissions for later review.
- Deepgram transcription (`nova-2`, `en-US`, punctuation off, diarization off) with per-word confidence.
- Inexact token alignment (Needleman–Wunsch style) between prompt and recognized words.
- Word-level stress inference from PocketSphinx phoneme alignment and pronunciation overlap scoring.
- Confidence visualization based on `confidence_cubed = confidence ** 3` as word **background color only**.
- A2A-compatible remote agent interface (Agent Card discovery + JSON-RPC endpoint) so other agents/platforms can call it.
  - `GET /.well-known/agent-card.json`
  - `POST /a2a` (`agent.about`, `pronunciation.evaluate`)
- Production-minded observability: request/trace IDs propagated through requests, responses, and logs for run correlation.
- Health endpoint (/healthz) to support deployment/monitoring and “is it alive?” checks.
- Structured, machine-consumable outputs (clear JSON schema for words, alignments, target evaluations, and summary metrics).
- "Control-plane friendly" service boundaries: stable HTTP endpoints + discoverable capabilities designed to plug into orchestration/routing.
- UI as an operator surface: single-page workflow that makes the agent capability usable and demoable (not just a script).
- Built-in developer visibility: the app page documents the agent endpoints and example calls to ease integration and adoption.
- Synchronous persistence of every analyzed submission as two files in a mounted bucket directory (default `/bucket`):
  - `YYMMDDHHMMSSffffff.wav`
  - `YYMMDDHHMMSSffffff.json`
  where the timestamp uses **HST** (`Pacific/Honolulu`) microseconds.

## Local setup

1. Make sure the `DEEPGRAM_API_KEY` environment variable is set. (It is in the Cloud Run configuration. We no longer use an `.env` file because we hope to make this repo public.)

2. (Optional, but recommended for persistence testing) set a custom bucket mount directory:

```bash
export BUCKET_DIR=/bucket
```

3. Run the dev server (creates `.venv`, installs deps, minifies assets, starts Flask):

```bash
./devserver.sh
```

4. Open:

```text
http://localhost:8080
```

## API endpoints

- `GET /api/paragraphs`
- `POST /api/analyze` (`multipart/form-data`: `paragraph_id`, `audio_wav`, optional `native_exemplar`)
- `GET /healthz`
- `GET /.well-known/agent-card.json`
- `POST /a2a`

### Persistence behavior (`/api/analyze` and `/a2a`)

For each successful analysis, the app writes two files synchronously into `BUCKET_DIR` (default `/bucket`):

- `{recording_id}.wav` (original uploaded WAV bytes)
- `{recording_id}.json` (analysis sidecar)

`recording_id` is generated from HST wall-clock time with microseconds:

```text
YYMMDDHHMMSSffffff
```

Example:

```text
/bucket/260226140321123456.wav
/bucket/260226140321123456.json
```

### Sidecar JSON schema (`schema_version = 1`)

Each sidecar file contains the following top-level structure:

```json
{
  "schema_version": 1,
  "recording_id": "YYMMDDHHMMSSffffff",
  "created_at_hst": "2026-02-26T14:03:21.123456-10:00",
  "timezone": "Pacific/Honolulu",
  "source": "web",
  "request_id": "uuid-or-upstream-id",
  "request_ip": "198.51.100.23",
  "paragraph_id": 3,
  "paragraph_text_hash": "sha256:...",
  "native_exemplar": true,
  "audio": {
    "path": "/bucket/YYMMDDHHMMSSffffff.wav",
    "bytes": 482344,
    "content_type": "audio/wav",
    "sample_rate_hz": 16000,
    "channels": 1,
    "sample_width_bytes": 2
  },
  "analysis_summary": {
    "percent_correct": 71.43,
    "total_targets": 7,
    "scored_targets": 7,
    "missing_targets": 0,
    "unaligned_targets": 0
  },
  "targets": [
    {
      "token_index": 12,
      "word_display": "record",
      "word_norm": "record",
      "label": "N",
      "expected_stress": 1,
      "inferred_stress": 2,
      "status": "ok",
      "correct": false,
      "core_phones": {"syll1": "EH", "syll2": "AO"},
      "core_durations": {"syll1": 0.07, "syll2": 0.11},
      "duration_ratio": 0.636364,
      "duration_ratio_log": -0.451985,
      "deepgram_word_index": 13,
      "deepgram_confidence": 0.93,
      "deepgram_confidence_cubed": 0.804357,
      "feedback": "Shift stress to syllable 1 and lengthen that vowel."
    }
  ],
  "pipeline": {
    "asr_provider": "deepgram",
    "asr_model": "nova-2",
    "aligner": "pocketsphinx"
  }
}
```

`targets[*].duration_ratio` is computed as `syll1/syll2` when both values exist and `syll2 > 0`; otherwise it is `null`.

`targets[*].duration_ratio_log` is computed as `ln((syll1 + 1e-4)/(syll2 + 1e-4))` when both values exist; otherwise it is `null`.

The sidecar JSON is written with an atomic temp-file-and-rename pattern (`.json.tmp` then `os.replace`) to reduce partial-write risk.

## Adaptive thresholding from native exemplar sidecars

The inference pipeline can learn stress decision thresholds from previously persisted sidecars in `$BUCKET_DIR/*.json`.

- **Data source:** all JSON sidecars in `BUCKET_DIR` (default `/bucket`).
- **Training filter:** only samples where top-level `native_exemplar` is `true`, target `status` is `"ok"`, `expected_stress` is 1 or 2, and target syllable durations are usable.
- **Feature:** `duration_ratio_log = ln((syll1 + 1e-4)/(syll2 + 1e-4))`.
- **Threshold method:** for each key, compute `thr = (median(class1) + median(class2)) / 2` where class1 is expected stress 1 and class2 is expected stress 2.
- **Keys:**
  - context key: `(word_norm, paragraph_id, token_index)` when available,
  - fallback key: `word_norm`.
- **Minimum data guardrail:** learned thresholds are used only when **both** classes have at least 2 exemplar samples.
- **Fallback behavior:** if no usable learned threshold exists (or bucket data is empty/corrupt), the app keeps the original heuristic: `syll1 >= syll2 => stress 1 else stress 2`.
- **Caching:** learned thresholds are cached in-process and refreshed periodically (TTL).

Target output now includes debug fields:

- `decision_method` (`"learned_threshold"` or `"naive_duration"` when inference succeeds)
- `duration_ratio_log`
- `learned_threshold` (nullable)
- `threshold_key` (nullable)

## Manual validation flow

1. Select a paragraph.
2. Click **Start Recording**, read paragraph, click **Stop Recording**.
3. Optionally check **native exemplar** if this submission should be flagged as an exemplar candidate.
4. Click **Submit for Analysis**.
5. Verify:
   - confidence-colored word backgrounds,
   - prominent red boxes around incorrect stress targets,
   - dashed orange boxes for missing/unaligned targets,
   - per-target table populated,
   - Developer/A2A docs visible on same page,
   - matching `{recording_id}.wav` and `{recording_id}.json` files appear in `BUCKET_DIR`.

## Tests

Run unit tests:

```bash
pytest -q
```

Included tests cover:
- paragraph parsing and target extraction,
- sequence alignment behavior,
- confidence-cubed + deterministic background normalization,
- persistence of HST sidecar+WAV files and schema fields using the paragraph 3 test WAV fixture.

## Cloud Run notes (high-level)

- Keep secrets out of source control.
- Provide `DEEPGRAM_API_KEY` via environment configuration.
- `Procfile` is already present for gunicorn start command and uses `app:app` (entry point: `app.py`).
- Validate locally with `devserver.sh` before deployment.
