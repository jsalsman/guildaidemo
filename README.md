# 

[![Try on Cloud Run](https://img.shields.io/badge/Try_on_Cloud_Run-darkgreen)](https://guildaidemo.talknicer.com/)
[![Agent health](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fguildaidemo.talknicer.com%2Fapi%2Fhealthz&query=%24.status&label=Agent%20health&color=brightgreen&labelColor=indigo)](https://guildaidemo.talknicer.com/api/healthz)
[![Python version 3.13](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://www.python.org/downloads/)
[![Flask version 3.1](https://img.shields.io/badge/Flask-3.1-black?logo=flask)](https://flask.palletsprojects.com/)
[![A2A Compatible](https://img.shields.io/badge/A2A-compatible-purple)](https://a2aprotocol.ai/)
[![MIT License](https://img.shields.io/badge/License-MIT-brightgreen)](https://opensource.org/licenses/MIT)
[![Donate](https://img.shields.io/badge/Donate-gold?logo=paypal)](https://paypal.me/jsalsman)

Single-page EFL pronunciation evaluator for noun/verb stress-shift pairs, with A2A-compatible discovery + JSON-RPC endpoints.

Try it: https://guildaidemo.talknicer.com

## Features

- Paragraph selector for all annotated paragraphs loaded from `PARAGRAPHS.txt` (currently 10, but dynamic).
- Browser microphone recording with WAV encoding.
- Optional "Bring Your Own Deepgram API Key" UI section that stores a user-supplied key in a `deepgram_api_key` browser cookie (365-day expiry).
- "native exemplar" checkbox in the UI to mark exemplar-candidate submissions for later review.
- Deepgram transcription with per-word confidence.
- Inexact token alignment (Needleman–Wunsch style) between prompt and recognized words.
- Word-level stress inference from PocketSphinx phoneme alignment and pronunciation overlap scoring.
- Confidence visualization based on `confidence_cubed = confidence ** 3` as background color.
- A2A-compatible remote agent interface (Agent Card discovery + JSON-RPC endpoint) so other agents/platforms can call it.
  - `GET /.well-known/agent.json`
  - `POST /a2a` (`agent.about`, `paragraphs.count`, `paragraphs.get_text`, `pronunciation.evaluate` with optional `deepgram_api_key`)
- Production-minded observability: request/trace IDs propagated through requests, responses, and logs for run correlation.
- Health endpoint (/healthz) to support deployment/monitoring and “is it alive?” checks.
- Structured, machine-consumable outputs (clear JSON schema for words, alignments, target evaluations, and summary metrics).
- "Control-plane friendly" service boundaries: stable HTTP endpoints + discoverable capabilities designed to plug into orchestration/routing.
- UI as an operator surface: single-page workflow that makes the agent capability usable and demoable (not just a script).
- Built-in developer visibility: the app page documents the agent endpoints and example calls to ease integration and adoption.
- Synchronous persistence of every analyzed submission as two files in a mounted bucket directory.

## Local setup

1. Optionally set `DEEPGRAM_API_KEY` as a shared fallback key for server-side transcription. The app now prefers a non-empty `deepgram_api_key` cookie when present and falls back to this environment variable when needed.

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
- `GET /api/healthz` (alias for environments that only route `/api/*`)
- `GET /.well-known/agent.json`
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
curl -s "$BASE_URL/.well-known/agent.json" | jq .
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
jq -n --arg audio "$AUDIO_B64" --arg dg "dg_live_xxx" '{jsonrpc:"2.0",id:"p3-a2a-demo",method:"pronunciation.evaluate",params:{paragraph_id:3,audio_wav_base64:$audio,deepgram_api_key:$dg}}' \\
| curl -s -X POST "$BASE_URL/a2a" \
    -H "Content-Type: application/json" \
    -d @- | jq .
```


### Deepgram API key resolution order

For requests that need transcription, the app resolves the Deepgram key in this order:

1. `deepgram_api_key` A2A method parameter (if provided and non-empty)
2. `deepgram_api_key` browser cookie (if provided and non-empty)
3. `DEEPGRAM_API_KEY` environment variable

Server logs include only the source (`a2a_param`, `cookie`, or `env`) and never print key values.

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
- confidence-cubed and deterministic background normalization,
- persistence of sidecar and WAV files and schema fields using the paragraph 3 test WAV fixture.

## Cloud Run notes

- Provide `DEEPGRAM_API_KEY` via the service environment configuration. Keep secrets out of source control.
- `Procfile` for Google Cloud Run invocations runs `app.py` as `app:app`.
- Validate locally with `devserver.sh` before deployment.

## Idea for multi-agent orchistration: WhatsApp Voice Message Bot via n8n

This section describes how to connect the pronunciation evaluator to WhatsApp so users can receive a practice paragraph, record a voice message reply, and get evaluation feedback — all within WhatsApp.

### Architecture overview

A lightweight n8n workflow acts as the orchestrator:

1. A user sends any text message to your WhatsApp number → n8n sends back a random paragraph (e.g., "Please read paragraph 3 aloud and reply with a voice message: ...")
2. The user replies with a WhatsApp voice message → n8n downloads the OGG/Opus audio
3. n8n calls a small audio conversion helper to produce a 16 kHz mono WAV
4. n8n base64-encodes the WAV and POSTs it to this app's `pronunciation.evaluate` A2A endpoint with the paragraph_id parsed from the outbound message text
5. n8n formats the JSON result into a plain-language summary and sends it back to the user

No server-side session state is required: the paragraph number is embedded in the text message sent to the user and parsed from the Meta webhook payload's conversation context when the voice reply arrives.

### Prerequisites

#### Meta developer account and WhatsApp Business Cloud API

WhatsApp's API is gated by Meta regardless of what orchestration layer you use. You will need:

- A **Meta developer account** at [developers.facebook.com](https://developers.facebook.com)
- A **Meta Business portfolio** (formerly Business Manager), verified with a real business or individual identity
- A **WhatsApp Business Cloud API app** created in the Meta developer console, with a verified phone number attached

Meta's verification process typically takes **1–3 business days** for individual/small business accounts, though it can be faster. The WhatsApp Cloud API itself is free at low message volumes (Meta's pricing is per-conversation, with a free tier for developer testing).

n8n's WhatsApp Business Cloud trigger node registers your webhook with Meta automatically using an OAuth token — you do not need to copy webhook URLs manually — but the underlying Meta account and app setup is still required.

#### n8n

Either:
- **Self-hosted** (recommended for this use case): `docker compose up` with the official n8n Docker image. Self-hosting gives you shell access for audio conversion and no workflow execution limits. Running n8n on Cloud Run or a small GCP VM keeps everything in the same GCP environment as the main app.
- **n8n Cloud**: easier to start but restricts arbitrary command execution, which affects audio conversion (see below).

### Audio format conversion

WhatsApp voice messages are delivered as OGG/Opus files. The `pronunciation.evaluate` endpoint requires 16 kHz mono WAV. Convert with ffmpeg:
```bash
ffmpeg -i input.ogg -ar 16000 -ac 1 -sample_fmt s16 output.wav
```

On self-hosted n8n you can run this in an Execute Command node. On n8n Cloud, or to keep the conversion cleanly separable, deploy a minimal Cloud Run function that accepts a POST with the OGG bytes and returns WAV bytes. This converter is itself a small A2A-compatible service if you want to expose it as a discoverable agent capability.

### n8n workflow outline

1. **WhatsApp Trigger node** — fires on any incoming WhatsApp message to your business number
2. **IF node** — branch on message type:
   - Text message → pick a random paragraph_id, fetch its text via `paragraphs.get_text`, send back "Please read paragraph {N} aloud and reply with a voice message: {text}"
   - Voice message → proceed to conversion and evaluation
3. **HTTP Request node** — download the OGG audio from the Media URL in the webhook payload
4. **Audio conversion** — Execute Command node (self-hosted) or HTTP Request to your conversion microservice
5. **Code node** — base64-encode the WAV bytes; parse the paragraph_id from the last outbound message text in the webhook context
6. **HTTP Request node** — POST to `/a2a`:
```json
   {
     "jsonrpc": "2.0",
     "id": "whatsapp-eval",
     "method": "pronunciation.evaluate",
     "params": {
       "paragraph_id": 3,
       "audio_wav_base64": "<base64>"
     }
   }
```
7. **Code node** — format result: extract `score_summary.percent_correct` and per-target feedback strings
8. **WhatsApp Business Cloud send node** — reply with the formatted summary

### Example summary message
```
Results for paragraph 3: 5/7 correct (71%)
✓ record (verb) — stress correct
✓ permit (verb) — stress correct
✗ project (noun) — should be PRO-ject; you stressed pro-JECT
✗ object (noun) — should be OB-ject; you stressed ob-JECT
✓ present (verb) — stress correct
✓ conduct (noun) — stress correct
✗ increase (noun) — should be IN-crease; you stressed in-CREASE
```

### Resources

- [Meta WhatsApp Cloud API getting started](https://developers.facebook.com/docs/whatsapp/cloud-api/get-started)
- [n8n WhatsApp Business Cloud trigger docs](https://docs.n8n.io/integrations/builtin/trigger-nodes/n8n-nodes-base.whatsapptrigger/)
- [n8n community WhatsApp voice message workflow template](https://n8n.io/workflows/3586-ai-powered-whatsapp-chatbot-for-text-voice-images-and-pdfs-with-memory/)
- [n8n self-hosting with Docker](https://docs.n8n.io/hosting/installation/docker/)

## Guild.ai proposal draft:

**The Oath:**
The Native English Speaker Homograph Stress Exemplar Crowdsourcer is an agent that
recruits native English speakers via Prolific to record themselves reading paragraphs
containing noun/verb homograph pairs, then submits those recordings to the Syllable
Stress Assessment Agent as native exemplars. Its purpose is to bootstrap the
data-driven stress-inference calibration of that backend, bringing threshold accuracy
from a naive duration heuristic to approximately 95% correct — the practical ceiling
given natural within-speaker variability. Without sufficient native exemplar data the
Syllable Stress Assessment Agent falls back to a simple "longer syllable wins" heuristic;
this agent exists to replace that fallback with statistically grounded, learned
thresholds for all 69 target homograph pairs.

**The Reagents:**
The agent is implemented in TypeScript and deployed on the Guild platform. It depends on
the Prolific API to create and monitor a study, recruit participants, and retrieve
completed submission metadata. Each participant is presented with one of ten paragraphs
covering all 69 target noun/verb homograph pairs as both parts of speech, and records
themselves reading it aloud via a browser-based interface. Completed audio submissions
are forwarded to the existing Syllable Stress Assessment Agent — a live A2A-compatible
Python backend at guildaidemo.talknicer.com on Google Cloud Run — via its
`pronunciation.evaluate` JSON-RPC endpoint with `native_exemplar: true`, which persists
each WAV and analysis sidecar to Google Cloud Storage and folds the new data into the
backend's adaptive threshold computation. Prolific participant fees for approximately 300
recordings are estimated at $200.

**The Ritual:**
The agent launches a Prolific study presenting participants with an assigned paragraph
and a recording interface; on submission it forwards the audio to the Syllable Stress
Assessment Agent's `pronunciation.evaluate` endpoint flagged as a native exemplar. As
recordings accumulate in GCS, the backend automatically recomputes per-word stress
decision thresholds from the growing distribution of syllable nucleus duration ratios.
The agent polls the Prolific API for study progress and queries the backend's convergence
status after each batch, checking how many of the 69 target words have accumulated
sufficient exemplars in both noun and verb roles to activate a learned threshold. Once
all words have converged, the agent closes the Prolific study and emits a final
convergence report.

**The Proof:**
The agent has succeeded when all 69 target homograph pairs report `decision_method:
learned_threshold` in the Syllable Stress Assessment Agent's evaluation responses,
indicating that naive duration fallback has been fully replaced by native-exemplar-
derived inference. The headline before/after metric is `percent_correct` on a held-out
validation set of test fixture WAVs replayed against the backend before the Prolific
study begins and again after convergence, quantifying the accuracy improvement the
crowdsourced exemplar data delivered. Study completion and per-word convergence progress
are themselves exposed as observable agent state, making the crowdsourcing pipeline
inspectable and steerable throughout its run.
