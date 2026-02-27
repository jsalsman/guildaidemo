# syllable-stress

Single-page EFL pronunciation evaluator for noun/verb stress-shift pairs, with A2A-compatible discovery + JSON-RPC endpoints.

Try it: https://guildaidemo.talknicer.com

## Features

- Paragraph selector for all annotated paragraphs loaded from `5-paragraph-syllable-stress-test_NV.txt` (currently 5, but dynamic).
- Browser microphone recording with WAV encoding.
- "native exemplar" checkbox in the UI to mark exemplar-candidate submissions for later review.
- Deepgram transcription with per-word confidence.
- Inexact token alignment (Needleman–Wunsch style) between prompt and recognized words.
- Word-level stress inference from PocketSphinx phoneme alignment and pronunciation overlap scoring.
- Confidence visualization based on `confidence_cubed = confidence ** 3` as background color.
- A2A-compatible remote agent interface (Agent Card discovery + JSON-RPC endpoint) so other agents/platforms can call it.
  - `GET /.well-known/agent-card.json`
  - `POST /a2a` (`agent.about`, `paragraphs.count`, `paragraphs.get_text`, `pronunciation.evaluate`)
- Production-minded observability: request/trace IDs propagated through requests, responses, and logs for run correlation.
- Health endpoint (/healthz) to support deployment/monitoring and “is it alive?” checks.
- Structured, machine-consumable outputs (clear JSON schema for words, alignments, target evaluations, and summary metrics).
- "Control-plane friendly" service boundaries: stable HTTP endpoints + discoverable capabilities designed to plug into orchestration/routing.
- UI as an operator surface: single-page workflow that makes the agent capability usable and demoable (not just a script).
- Built-in developer visibility: the app page documents the agent endpoints and example calls to ease integration and adoption.
- Synchronous persistence of every analyzed submission as two files in a mounted bucket directory.

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


### Playwright screenshot troubleshooting

If `mcp__browser_tools__run_playwright_script` fails with `net::ERR_EMPTY_RESPONSE`, the usual cause is that no web server is actually listening yet on `localhost:8080`.

Use this sequence before running Playwright:

```bash
export DEEPGRAM_API_KEY=dummy
python app.py
```

Then run the Playwright script against `http://localhost:8080` (or `127.0.0.1:8080`).

Why this helps:
- The app refuses to start when `DEEPGRAM_API_KEY` is unset.
- Playwright only captures pages from forwarded ports that already have an active listener.

## API endpoints

- `GET /api/paragraphs`
- `POST /api/analyze` (`multipart/form-data`: `paragraph_id`, `audio_wav`, optional `native_exemplar`)
- `GET /healthz`
- `GET /.well-known/agent-card.json`
- `POST /a2a`

Agent card discovery advertises method-level capabilities for `agent.about`, `paragraphs.count`, `paragraphs.get_text`, and `pronunciation.evaluate`, including required vs optional params and the JSON-RPC endpoint URL.

### Persistence behavior (`/api/analyze` and `/a2a`)

For each successful analysis, the app writes two files synchronously into `BUCKET_DIR`: `{recording_id}.wav` (original uploaded WAV bytes) and `{recording_id}.json` (analysis sidecar). The `recording_id` is a microsecond-resolution timestamp string. Example:

```text
/bucket/260226140321123456.wav
/bucket/260226140321123456.json
```

### Recording JSON schema

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

Target output now includes debug fields:

- `decision_method` (`"learned_threshold"` or `"naive_duration"` when inference succeeds)
- `duration_ratio_log`
- `learned_threshold` (nullable)
- `threshold_key` (nullable)

### A2A JSON-RPC quickstart (paragraph 3 WAV fixture)

1. Read the discoverable model card:

```bash
curl -s "$BASE_URL/.well-known/agent-card.json" | jq .
```

2. Build base64 payload from the paragraph 3 WAV fixture:

```bash
AUDIO_B64=$(python - <<'PY'
import base64
from pathlib import Path
print(base64.b64encode(Path("tests/abc340c7-fc39-41f0-b1a6-3557f83b7707.wav").read_bytes()).decode())
PY
)
```


3. Discover paragraph count before selecting ids:

```bash
curl -s -X POST "$BASE_URL/a2a" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"p-count","method":"paragraphs.count","params":{}}' | jq .
```

4. Fetch plain paragraph text (unannotated) for the selected id:

```bash
curl -s -X POST "$BASE_URL/a2a" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"p-text-3","method":"paragraphs.get_text","params":{"paragraph_id":3}}' | jq .
```

5. Submit `pronunciation.evaluate` as an A2A client:

```bash
jq -n --arg audio "$AUDIO_B64" '{jsonrpc:"2.0",id:"p3-a2a-demo",method:"pronunciation.evaluate",params:{paragraph_id:3,audio_wav_base64:$audio}}' \
| curl -s -X POST "$BASE_URL/a2a" \
    -H "Content-Type: application/json" \
    -d @- | jq .
```

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
- `Procfile` for Google Cloud Run invocations runs `app.py` as `app:app`.
- Validate locally with `devserver.sh` before deployment.
