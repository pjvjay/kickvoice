# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- **Intel Macs:** `transcribe`, `debate`, and the MCP server behind `monitor`
  and `extract-profiles` aborted with `OMP: Error #15` once both the voice and
  the speech model were loaded, unless `KMP_DUPLICATE_LIB_OK=TRUE` was exported
  by hand, and even then crashed or hung when the voice model was used first.
  kickvoice now arranges a working setup itself on macOS x86_64: it sets the
  switch, loads CTranslate2 before PyTorch and keeps PyTorch on one thread (see
  "Intel Macs" in the README, including the caveat and how to opt out).
- When the MCP server process dies, the terminal now shows the end of its log
  instead of only `MCPError: Connection closed`.
- `monitor` stayed alive after printing its report until Enter was pressed: a
  thread was left blocked reading the terminal. Input is now read in short
  polls, so the process exits when the run ends (also after Ctrl+C while
  `extract-profiles` is waiting for an answer).
- Fact-checking missed two kinds of misquote. Made-up quotes in single or
  curly single quote marks (also `«…»` and `„…“`) were never checked, and
  `...` could drop a word, so "are … fine" passed as a quote of "are not
  fine". All quote marks are checked now, and a quote must be one unbroken
  passage.
- The install command in the README and the project URLs in the package
  metadata pointed at a GitHub repository that does not exist. They now point
  at this one.

### Changed
- `...` no longer joins separate fragments inside a quote; such a reply is
  rejected with an explanation. Quote one passage, or paraphrase without
  quote marks.

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
