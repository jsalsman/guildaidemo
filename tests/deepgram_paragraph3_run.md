# Deepgram Paragraph 3 Connectivity + Verbose Debug Run

- Date: 2026-02-25
- Command: `pytest -s tests/test_paragraph3_verbose_debug.py`
- Environment check: `DEEPGRAM_API_KEY` was present in environment.

## Result

- Deepgram request succeeded and returned timed words for the uploaded WAV.
- Test finished successfully: `1 passed`.
- Network path to Deepgram is working in this environment.

## Notable debug observations from test output

- Deepgram transcript token was `affect` while paragraph token is `effect`, leaving `effect` mapped as `missing`.
- Most other paragraph-3 targets were mapped and produced inferred stress (`status: ok`).
- Several inferred stresses still differ from expected values (e.g., `desert`, `essay`, `extract`, `increase`, `impact`).

## Next debugging focus

1. Add lexical-variant handling for likely homophone substitutions (`effect`/`affect`) during alignment.
2. Investigate stress inference disagreements where alignment exists and durations are populated.
