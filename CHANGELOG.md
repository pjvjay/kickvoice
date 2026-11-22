# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.0]

### Added
- **Agent commands.** `extract-profiles` and `monitor` now spawn goal-driven
  AI agents that work through kickvoice's MCP server, which runs as a
  separate process over stdio.
  - `extract-profiles`: listens, identifies each distinct voice (transcript
    clues, channel, Wikidata), exports voice clips, and creates profiles only
    with the user's approval.
  - `monitor`: a debate copilot. It follows the stream, recognises the
    opponent by voice, and proposes chat replies that are fact-checked before
    they're shown. A typed number copies a reply. Nothing is posted to chat.
  - Policies: `--policy llm` (default; Anthropic or any OpenAI-compatible
    model, including local ones), `--policy rules` (deterministic, no model),
    and `--no-agent` (the original pipelines).
  - Every run writes `trace.jsonl` (every model call, tool call, question and
    decision, written live), `report.md`/`report.json`, the exact
    `config.json`, and the MCP server log.
  - Bounded episodes (step cap, timeout) and a whole-run token budget
    (`--max-tokens`).
- **MCP server** (`kickvoice mcp`): 22 tools for live sessions, voice
  profiling, research and reply drafting, plus a `live_chat_copilot` prompt.
  `export_voice_clip` and `start_live_session(keep_audio=...)` support
  listening before approving.
- **Speaker-attributed transcription** (`transcribe`, `transcripts …`).
- **Opposition research** (`research …`), with a live `debate` mode.
- Profile extraction from streams, profile packs (export/import).
- Packaging: `src/` layout, PEP 621/639 metadata, `py.typed`, extras (`all`,
  `agent`, `transcribe`, `youtube`, `dev`), `MANIFEST.in`.
- Quality gates: ruff lint and format, pytest with branch coverage (≥90%
  enforced, including spawned MCP subprocesses), and GitHub Actions CI.

### Fixed
- `run.sh` didn't put the virtualenv on `PATH`, so `streamlink` and `yt-dlp`
  weren't found.
- `enroll --youtube` downloaded every video twice.
- `make verify` and `make download-models` failed with an `IndentationError`.
- Offline streams reported an empty streamlink error.
- Twitch VODs were stored as YouTube videos, with broken links.
- Fact-checking accepted quotes with one word changed; it now requires an
  exact match.
- Jokes flagged as insincere could still be quoted back via their raw excerpt.
- In `monitor`, a reply to the opponent's last argument could be cancelled
  when the stream ended.

### Changed
- `monitor` and `extract-profiles` default to agent mode. Use `--no-agent`
  for the previous behaviour.
- `torchaudio` is no longer a direct dependency (kickvoice never imports it;
  SpeechBrain declares it).

## [0.1.0]

- Initial release: enrolment from files/YouTube, live speaker identification.
