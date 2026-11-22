# Contributing

## Setup

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[all,dev]"
```

## Before opening a pull request

```bash
make lint        # ruff check (and `ruff format --check` in CI)
make coverage    # pytest with branch coverage; fails under 90%
make check-dist  # builds the sdist/wheel and runs twine check
```

CI runs the same checks on Python 3.10–3.12.

## Tests

- Tests must not need ML models or network access. Use the fakes in
  `tests/conftest.py`:
  - `FakeEncoder`, `FakeRecognizer`, `SlowCapture`: synthetic voices are
    pitched tones.
  - `FakeWeb`: an HTTP server for Wikipedia, X, Reddit and model APIs.
  - `ScriptedLLM` / `ScriptedToolLLM`: scripted models. Tool-calling scripts
    decide each step from earlier tool results, as a real agent would.
  - Fake executables on `PATH` for `streamlink` / `yt-dlp`.
- Prefer exercising real code paths (real subprocesses, real MCP connections,
  real SQLite) over mocking internals.
- Mark tests that launch whole CLI processes with `@pytest.mark.slow`.

## Design rules worth keeping

- **Nothing acts on a person without approval.** Profiles, identity links and
  paid analysis go through an explicit user confirmation.
- **Quotes are verbatim or they aren't quotes.** Anything shown as someone's
  words is checked against the stored source text.
- **Agents don't get the last word on accuracy.** The orchestrator verifies
  replies itself (`verify_message`) before showing them.
- **Every agent run is fully traced.** If you add a step, trace it.
