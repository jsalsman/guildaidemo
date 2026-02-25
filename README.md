# syllable-stress

Single-page EFL pronunciation evaluator for noun/verb stress-shift pairs, with A2A-compatible discovery + JSON-RPC endpoints.

Try it: https://guildaidemo.talknicer.com

## Features

- Paragraph selector for 5 annotated paragraphs from `5-paragraph-syllable-stress-test_NV.txt`.
- Browser microphone recording with WAV encoding to **16kHz / 16-bit PCM / mono**.
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

## Local setup

1. Make sure the `DEEPGRAM_API_KEY` environment variable is set. (It is in the Cloud Run configuration. We no longer use an `.env` file because we hope to make this repo public.)

2. Run the dev server (creates `.venv`, installs deps, minifies assets, starts Flask):

```bash
./devserver.sh
```

3. Open:

```text
http://localhost:8080
```

## API endpoints

- `GET /api/paragraphs`
- `POST /api/analyze` (`multipart/form-data`: `paragraph_id`, `audio_wav`)
- `GET /healthz`
- `GET /.well-known/agent-card.json`
- `POST /a2a`

## Manual validation flow

1. Select a paragraph.
2. Click **Start Recording**, read paragraph, click **Stop Recording**.
3. Click **Submit for Analysis**.
4. Verify:
   - confidence-colored word backgrounds,
   - prominent red boxes around incorrect stress targets,
   - dashed orange boxes for missing/unaligned targets,
   - per-target table populated,
   - Developer/A2A docs visible on same page.

## Tests

Run unit tests:

```bash
pytest -q
```

Included tests cover:
- paragraph parsing and target extraction,
- sequence alignment behavior,
- confidence-cubed + deterministic background normalization.

## Cloud Run notes (high-level)

- Keep secrets out of source control.
- Provide `DEEPGRAM_API_KEY` via environment configuration.
- `Procfile` is already present for gunicorn start command.
- Validate locally with `devserver.sh` before deployment.

## License

Open source, MIT License. By Jim Salsman, February 25, 2026.

