# kickvoice

**Speaker identification, transcription, opposition research and an AI debate
copilot for Kick.com live streams.**

- **Who's speaking:** voice profiles built with ECAPA-TDNN (SpeechBrain), matched live.
- **What they said:** speaker-attributed transcripts from Whisper.
- **What they've said before:** a cited, fact-checked record of a public
  figure's public statements.
- **Agents that do the work:** `extract-profiles` and `monitor` spawn goal-driven
  AI agents that drive all of the above through kickvoice's own
  [MCP](https://modelcontextprotocol.io) server. They write reports and full
  step-by-step traces.

> ⚠️ **Not Dejavu.** This is speaker *recognition* (identify a person by voice),
> not audio *fingerprinting* (match a specific recording). See [Architecture](#architecture).

## Installation

**Requirements:**
- Python 3.10–3.12
- [ffmpeg](https://ffmpeg.org) (`brew install ffmpeg` / `sudo apt install ffmpeg`)
- For agents: an LLM API key (`ANTHROPIC_API_KEY` by default), or a local
  OpenAI-compatible model. Alternatively, run agents with `--policy rules`,
  which needs no model.

### Option A: pip (a standard Python package)

```bash
# 1. PyTorch first, from pytorch.org: pick the CPU or CUDA build for your machine.
#    Installing torch and torchaudio together keeps them on matching builds.
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. kickvoice with everything (agents, transcription, YouTube):
pip install "kickvoice[all] @ git+https://github.com/kickvoice/kickvoice"
# …or from a built wheel:  pip install "kickvoice-0.2.0-py3-none-any.whl[all]"

kickvoice --help
```

| Extra | Adds |
|---|---|
| *(none)* | Voice profiles, speaker ID, profile extraction, research |
| `transcribe` | Speaker-attributed transcription (faster-whisper) |
| `agent` | The MCP server and the spawned agents (`mcp`) |
| `youtube` | Enrolment from YouTube, YouTube/Twitch research (`yt-dlp`) |
| `all` | All of the above |
| `dev` | pytest, coverage, ruff, build, twine |

### Option B: the setup script (creates a virtualenv for you)

```bash
git clone <this-repo> && cd kickvoice
chmod +x setup.sh run.sh
./setup.sh          # asks CPU or GPU PyTorch, installs kickvoice[all,dev] into .venv
```

Afterwards `./run.sh <command>` runs kickvoice inside that environment;
`kickvoice <command>` works the same once the venv is activated. Non-interactive
alternatives: `make install`, `make install-gpu`, `make download-models`.

### Intel Macs (macOS x86_64)

Intel Macs are supported. Two things are different there:

- **Python 3.12 and PyTorch 2.2.2.** PyTorch 2.2.2 is the last release with
  Intel Mac wheels, and it has none for Python 3.13 or later. `./setup.sh` and
  `make install` pick both for you.
- **Two OpenMP runtimes in one process.** The Intel Mac wheels of PyTorch and of
  CTranslate2 (which faster-whisper runs on) each bundle their own copy of
  Intel's OpenMP runtime, `libiomp5`. As installed, loading both aborts:

  ```
  OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
  ```

  That stopped `transcribe`, `debate`, and the MCP server behind `monitor` and
  `extract-profiles`, where the terminal only showed `MCPError: Connection
  closed`.

  kickvoice now sets this up itself. You don't need to export anything,
  whether you use `./run.sh`, the `kickvoice` command or `python -m kickvoice`:

  1. It sets `KMP_DUPLICATE_LIB_OK=TRUE` in its own processes, including the
     MCP server it spawns, so the second runtime is allowed to start.
  2. It loads CTranslate2 before PyTorch and keeps PyTorch on one thread. With
     both runtimes loaded, the library that was loaded second crashes or hangs
     as soon as it uses several threads, and works on one. Whisper is the
     heavy model, so it keeps its threads; the voice models (ECAPA, Silero
     VAD) are light, and importing silero-vad already put PyTorch on one
     thread, so in practice this costs nothing.

  All of this happens only on macOS x86_64 with faster-whisper installed. The
  details and the measurements behind them are in `src/kickvoice/_openmp.py`.

  What to know:
  - Intel documents `KMP_DUPLICATE_LIB_OK` as an unsupported workaround and
    does not rule out crashes or wrong results. The arrangement above was
    tested on an Intel Mac with torch 2.2.2 and ctranslate2 4.9.0
    (`tests/test_openmp.py` runs both libraries for real on such a machine);
    other versions may behave differently.
  - If you would rather have the abort, export `KMP_DUPLICATE_LIB_OK=FALSE`
    before running kickvoice. A value you set yourself is never overridden.
  - If your own program imports `torch` before `kickvoice`, the roles swap:
    PyTorch keeps its threads and Whisper runs on one, which is slower. A
    warning is logged.
  - If the MCP server stops for any other reason, the terminal now shows the
    end of its log; the whole log is `mcp_server.log` in the run folder.

## The two agent commands

Both commands spawn an AI agent with a specific goal. The agent works only
through kickvoice's MCP tools, and the server is started as a separate
process (`kickvoice mcp`), exactly as Claude Desktop or Claude Code would
use it.

```bash
./run.sh extract-profiles --stream "https://kick.com/somechannel"
./run.sh monitor --stream "https://kick.com/somechannel" --opponent "Jane Pundit"
```

| | `extract-profiles` | `monitor` |
|---|---|---|
| **Goal** | Build voice profiles for everyone who speaks | Help you answer the opponent's arguments in chat |
| **How** | Listens, then identifies each distinct voice (what's said, the channel, Wikidata for public figures), exports a clip, asks you, enrolls approved voices | Follows the debate, recognises the opponent by voice, and for each argument drafts chat replies grounded in their record |
| **Your part** | Approve or name each voice (nothing is created without you) | Type a reply's number to copy it (nothing is posted for you) |
| **Outputs** | Report + voice clips + trace | Report (arguments, replies, sources) + trace |

**How an agent runs.** A lightweight orchestrator follows the stream over
MCP and wakes the agent only when there is something to decide. Each
decision is a short, bounded **episode** ("identify these voices", "answer
argument p3"), with a step cap, a timeout and a whole-run token budget. So a
three-hour stream doesn't become one ever-growing, expensive conversation.

```
run.sh monitor ─▶ run folder + config ─▶ spawns `kickvoice mcp` (stdio)
      │                                          ▲
      ▼                                          │ MCP tool calls
  orchestrator ── get_live_updates (long-poll) ──┤
      │ new argument / unknown voice             │
      ▼                                          │
  agent episode (model + tools: get_point_evidence, propose_replies, …)
      │ propose_replies
      ▼
  orchestrator fact-checks every reply (verify_message) ─▶ shown to you ─▶ "2" copies it
```

**Policies:**
- `--policy llm` (default): a model drives the agent. Set `ANTHROPIC_API_KEY`,
  or use an OpenAI-compatible server (see [Models](#7-research-an-opponent-then-get-live-counter-arguments)).
- `--policy rules`: a deterministic agent with no model. It uses the same MCP
  tools and writes the same traces and reports. `extract-profiles` asks you
  to name each voice; `monitor` offers "receipts", their own past words on
  the topic, verbatim and dated.
- `--no-agent`: the original pipelines, with no agent at all.

**Outputs.** Every run writes a folder under `~/.kickvoice/runs/`:

```
20250601-201500_monitor_somechannel/
  report.md / report.json   what the run produced
  trace.jsonl               every step, written live: model calls (with token counts),
                            tool calls + results, questions to you and your answers,
                            decisions, live transcript events, errors
  config.json               the exact settings used
  mcp_server.log            the MCP server's own log
  clips/                    voice clips for review (extract-profiles)
```

Useful options:
- `--duration S`: how long to listen.
- `--model M`: the model to use.
- `--max-tokens N`: token budget for the run (default 2M; 0 = no cap).
- `--opponent` / `--me` (monitor).
- `--min-speech S` (extract-profiles).

Agent settings live in the `agent` section of `config.json`.

### Validating it yourself

**No API key or live stream needed.** Point the commands at any recording
with two or more people talking:

```bash
./run.sh extract-profiles --file debate.mp4 --policy rules --min-speech 10
#   → name each voice when asked; check report.md, the clips, and `kickvoice profiles list`
./run.sh research add "Jane Pundit" --voice Jane --x JanePundit
./run.sh research collect "Jane Pundit" && ./run.sh research analyze "Jane Pundit"
./run.sh monitor --file debate.mp4 --opponent "Jane Pundit" --policy rules
#   → arguments appear with "receipts"; type a number to copy; read report.md + trace.jsonl
```

**With a model:** repeat the commands without `--policy rules`, then compare
the `trace.jsonl` of the two runs. It shows each agent decision and the tool
call behind it.

## Usage guide

### 1. Enroll speakers

Use the `run.sh` wrapper (auto-activates the venv), or activate manually:

```bash
# Option A: wrapper script (no manual venv activation needed)
./run.sh enroll "DrDisrespect" --file /path/to/audio.wav
./run.sh enroll "xQc" --file /path/to/clip.mp3

# Option B: activate venv manually
source .venv/bin/activate
kickvoice enroll "DrDisrespect" --file /path/to/audio.wav
```

From video URLs (requires `yt-dlp`, installed automatically with setup):

```bash
./run.sh enroll "DrDisrespect" \
  --youtube https://youtube.com/watch?v=ABC \
  --youtube https://youtube.com/watch?v=DEF
```

Files and URLs can be mixed, and every source is used once:

```bash
./run.sh enroll "xQc" --file calm.wav --file shouting.mp3 --youtube https://youtu.be/ABC
```

> **Tip:** Use 2-3 videos per speaker with different contexts (calm, excited,
> shouting) for more robust profiles. More source material = better accuracy.

### 2. Monitor a Kick.com stream (speaker ID only)

`monitor` spawns the debate-copilot agent (see above). For plain
who-is-speaking output, use the classic pipeline:

```bash
./run.sh monitor --stream "https://kick.com/somechannel" --no-agent
```

Output:
```
🎙  Monitoring stream with 2 enrolled speaker(s)
   Profiles: DrDisrespect, xQc
   Chunk: 10.0s, Stride: 5.0s
   Match threshold: 0.6

[14:32:07] DrDisrespect detected (confidence: 0.82)
[14:32:15] Unknown speaker (best: 0.34)
[14:32:28] xQc detected (confidence: 0.74)
```

### 3. Test with a local file

```bash
./run.sh monitor --file recording.wav --no-agent
```

### 4. Extract profiles without an agent (discover who's talking)

Don't have clean enrollment audio? Let kickvoice listen to a stream, work out
how many distinct voices there are, and build profiles from them, with you
approving every one. By default this command spawns the profile-extraction
agent (see above). `--no-agent` runs the original pipeline shown here:

```bash
./run.sh extract-profiles --stream "https://kick.com/somechannel" --no-agent
```

```
🎧 Listening to https://kick.com/somechannel (up to 5m 00s; Ctrl+C to stop early)
   Known profiles: 2

   [00:30] speech so far: 22s, 21 voice windows
   ...
⏹  Stopped listening after 5m 00s.
🔬 Grouping 212 voice windows by speaker…

🗣  Found 3 speaker(s) in 5m 00s of audio (3m 41s of speech)

   #  Speech    First heard  Best match
   1  2m 13s    00:04        ✅ DrDisrespect (0.82)
   2  51s       01:12        🆕 new voice (closest: xQc 0.31)
   3  24s       03:40        🆕 new voice (closest: DrDisrespect 0.18)
```

Each voice is then shown for approval, with a ~20s **review clip** of just
that voice so you can check who it is before anything is saved:

```
Speaker 2 of 3 · 51s of speech · first heard at 01:12
  Review clip: ~/.kickvoice/extractions/20250601-201500_somechannel/speaker_2.wav
  🆕 No match among 2 known profile(s) (closest: xQc 0.31)

  [n] Create a new profile for this voice
  [m] Merge into an existing profile
  [p] Play clip   [s] Skip   [q] Quit (finish later)
  Choice [s]:
```

- **Known voice** → `[a]` confirms the match and folds the new audio into
  that profile (weighted by speech time, so the profile gets more robust to
  stream audio without a short clip overwriting it).
- **New voice** → `[n]` asks for a name. For the most talkative voice on a
  `kick.com/<channel>` URL, the channel name is suggested (usually the
  streamer), but you still have to confirm it.
- **Enter = skip.** Nothing is written to your profiles without an explicit
  choice. Game audio, TTS donations and music that slip through as a
  "voice" can simply be skipped.
- **`[q]` or Ctrl+C** stops reviewing; undecided voices stay pending.
- Profiles approved earlier in the same review are matched immediately, so if
  one person was split into two groups, the second one shows up as a match.

```bash
# Options
./run.sh extract-profiles --stream URL --duration 900     # listen 15 min (0 = until Ctrl+C/end)
./run.sh extract-profiles --file vod.mp4                  # a downloaded VOD / recording
./run.sh extract-profiles --stream URL --min-speech 10    # keep voices with ≥10s speech (default 20)
./run.sh extract-profiles --stream URL --no-review        # just collect; approve later

# Review later / resume a partial review
./run.sh review                    # newest session with pending voices
./run.sh review --list             # all sessions
./run.sh review --session 20250601-201500
```

Ctrl+C while listening stops capture and continues with whatever was
collected. Sessions (voiceprints + review clips) live in
`~/.kickvoice/extractions/`; delete old ones whenever you like.

> **Privacy note:** voiceprints are biometric data, and streams often include
> guests and callers who aren't public figures. The approval step exists so
> that you decide who gets a profile. Check the rules where you live
> (e.g. GDPR, Illinois BIPA) before storing or sharing profiles of private
> individuals.

### 5. Manage and share profiles

```bash
./run.sh profiles list
./run.sh profiles info "DrDisrespect"
./run.sh profiles remove "xQc"          # asks for confirmation (-y to skip)

# Profile packs: portable JSON bundles of voiceprints
./run.sh profiles export --out streamers.json                  # all profiles
./run.sh profiles export --out two.json --name "xQc,Adin Ross"  # selected
./run.sh profiles import streamers.json                        # skip existing names
./run.sh profiles import streamers.json --on-conflict merge    # or: overwrite
```

A pack is a single diff-friendly JSON file (~3 KB per profile), which makes it
easy to keep a shared library of known streamer voices in a git repo and grow
it with `extract-profiles`. I'm not aware of any existing public repository
of streamer voiceprints, so packs are the way to build one.

Packs record the embedding model that produced them. Voiceprints from
different models aren't comparable, so importing a pack made with a
different model is refused.

### 6. Transcribe: who said what

```bash
./run.sh transcribe --stream "https://kick.com/somechannel"
./run.sh transcribe --file vod.mp4 --language en
```

```
[00:07] xQc: so the thing about sponsorships is that nobody reads the contract
[00:14] (unknown): chat is going crazy right now
[00:16] xQc?: yeah yeah
```

Every line is labelled with an **attribution level**:

| Level | Meaning | Shown as |
|---|---|---|
| `confident` | The voice and the line both match strongly, on enough audio (≥2s) | `xQc` |
| `probable` | Matched, but short, contested, or weaker on its own | `xQc?` |
| `unknown` | No enrolled voice matched | `(voice 3)` / `(unknown)` |

Only enrolled speakers are recognised. Unrecognised people still get a
**voice group** number, so you can tell two unknown guests apart.

**How attribution works.** Lines are labelled as they're transcribed. When
the session ends (or you press Ctrl+C), a **whole-session check** groups the
lines by voice, and each profile can claim at most one voice group. Then a
line is only "confident" if both its group *and* the line itself match
strongly. Both checks came out of testing on real multi-speaker audio:

- an un-enrolled guest on the same mic scored exactly like the host line by
  line. Only the one-group-per-profile rule kept them apart;
- two similar voices got grouped together, but their individual lines scored
  very differently. Only the per-line check caught that.

In that testing, every "confident" line was correct. The remaining mistakes
were all labelled "probable", so analysis should use confident lines only.

```bash
./run.sh transcripts list                                    # sessions
./run.sh transcripts show SESSION [--speaker xQc] [--min-attribution confident]
./run.sh transcripts search "gambling" --speaker xQc        # full-text search
./run.sh transcripts speakers                                # speech on record per person
./run.sh transcripts export --out xqc.jsonl --speaker xQc --min-attribution confident
./run.sh transcripts reattribute                             # after enrolling someone new
./run.sh transcripts delete SESSION
```

- **Transcribe now, enroll later.** Each line's voice embedding is stored (not
  the audio), so `transcripts reattribute` can relabel old transcripts after
  you enroll someone or fix a profile.
- **Export** writes JSON Lines, one line per utterance with `speaker`,
  `attribution`, `voice_group`, `start_s`/`end_s`, `said_at`, `source` and
  `text`. This is the input format for analysis.
- **Speed.** Whisper runs on CPU by default. The summary prints a real-time
  factor: if it's above 1.0 on a live stream, you'll fall behind. Use a smaller
  `--model` (`base`, `tiny`) or `--device cuda`. For long VODs, `--file` has
  no time pressure, so larger models are fine.
- **Whisper and silence.** Whisper tends to invent text over music and
  silence ("Thanks for watching!"). Words outside detected speech are
  dropped.
- **Privacy.** Unrecognised voices' text is kept as context (e.g. what the
  streamer is replying to). `--drop-unknown` stores only enrolled speakers'
  lines. Transcripts live in `~/.kickvoice/transcripts.db`.

**Limits.**
- Attribution can only separate voices the speaker model can separate. If
  two people sound alike *to the model* and talk over each other with no
  pauses, their words can end up in one line.
- Lines under ~1s (e.g. "yeah") are too short to embed and are rarely
  attributed.
- Transcription needs the `faster-whisper` package (installed by setup). The
  Whisper model downloads on first use.

### 7. Research an opponent, then get live counter-arguments

Before a debate, kickvoice builds a cited record of what your opponent has
said publicly. During the debate, it listens for their points and suggests
replies grounded in that record.

```bash
# 1. Who are they? Link accounts (yours are trusted; discovered ones need your OK)
./run.sh research add "Jane Pundit" --x JanePundit --youtube @janepundit --voice Jane
./run.sh research discover "Jane Pundit" --x JanePundit     # official accounts via Wikidata
./run.sh research discover "Jane Pundit" --wikidata Q12345  # …or name the exact record

# 2. Collect their public record
./run.sh research collect "Jane Pundit"                     # Wikipedia, YouTube captions, X, Reddit, site
./run.sh research collect "Jane Pundit" --voice --appearances 10   # + their words in other channels' videos

# 3. Extract their claims (LLM), then prep
./run.sh research analyze "Jane Pundit"
./run.sh research brief "Jane Pundit" --out brief.md        # positions, quotes, contradictions, counters
./run.sh research search "Jane Pundit" "rent control"

# 4. Live
./run.sh debate --stream "https://kick.com/somechannel" --opponent "Jane Pundit" --notes prep.md
```

```
[12:04] Jane: rent control protects tenants and keeps housing stable for families
┌─ 💡 Rent control keeps housing stable
│ ⚠ CONTRADICTION: In 2021 she argued it shrinks supply.
│    "Rent control reduces housing supply, build more homes" — 2021-03-01, their own words
│    https://x.com/JanePundit/status/1366…
│ ↩ 1. You said in 2021 that rent control shrinks supply — what changed?
│ ↩ 2. Stable for current tenants, but what about the people who can't find a unit?
│      └ your notes: SF 2019 study: supply of rent-controlled buildings fell 15%
└─ 2.1s · claude-haiku-4-5
```

**What it collects.** Public content only, through official APIs and
permitted access:

| Source | How | Needs |
|---|---|---|
| Wikipedia / Wikidata | MediaWiki API | — |
| YouTube (their channel) | captions via yt-dlp | — |
| YouTube (appearances elsewhere), Twitch VODs | audio + voice-matched transcription (`--voice`) | their voice profile |
| X | API v2 | `X_BEARER_TOKEN` (paid API tier) |
| Reddit | OAuth app | `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` |
| Their website, web articles | page fetch (robots.txt respected); search via Brave | `BRAVE_API_KEY` for search |
| Your stream transcripts | lines kickvoice attributed to their voice | their voice profile |
| Anything else you have | `research import NAME file.jsonl` (or an X data export) | — |

**Linking the right person.** `discover` only ever links records of
**people**. Games, companies, shows and fictional characters also have
official accounts on Wikidata (the video game *Destiny* owns @DestinyTheGame),
so those are skipped and reported. How it finds the record matters:

| Evidence | Example | `-y` auto-confirms? |
|---|---|---|
| A record lists a handle you gave | `--x TheOmniLiberal` | yes |
| The record id you gave | `--wikidata Q12345` | yes |
| Name search only | `discover "Destiny"` | **no**: you choose in a terminal, or it stops and lists the `--wikidata` ids |

If a handle you give isn't on Wikidata, it says so instead of silently
using a name match.

**Fixing a wrong link.** Every collected document remembers which account
it came from, and documents from a **rejected** account are ignored from
then on.

```bash
./run.sh research discover "Destiny" --wikidata Q12345 -y   # re-link: the old record's accounts are rejected
./run.sh research unlink "Destiny" --purge                  # or undo the link entirely
./run.sh research accounts "Destiny" --reject 7 --purge     # reject one account and delete its data
```

`research show` warns when data from rejected accounts is still stored.

**What it doesn't do:**
- **Discord.** There's no public API for a user's messages, and scraping
  them breaks Discord's terms.
- **Private or locked accounts, login walls, or alt-account unmasking.**
  Accounts are linked only when you add them or they're publicly listed on
  Wikidata, and you confirm each one.
- **Personal life.** Claim extraction ignores family, health and location.

**Accuracy rules** (quoting someone wrong in a live debate backfires):

- **Authorship is recorded for everything.** Each item is marked as *their
  own words*, *their voice (verified)*, *their channel (speaker
  unverified)*, or *reported about them*. Contradictions can only be built
  from their own words.
- **Quotes are checked against the source.** A quote the model can't back
  up with the source text is dropped. The stored quote is the source's
  words, never the model's paraphrase.
- **Jokes and sarcasm are flagged**, so they aren't thrown back as serious
  positions.
- **Live suggestions can only cite evidence they were given.** Citations to
  anything else are discarded. A suggestion with no source is labelled
  "reasoning, no source".
- **Live lines are provisional.** The opponent's lines are detected by
  voice as they're transcribed, so the confident/probable labels are
  per-line. The whole-session check runs after the debate.

**Models.** Set `ANTHROPIC_API_KEY` (default provider), or use any
OpenAI-compatible server (including local models through Ollama):

```json
{"llm": {"provider": "openai", "base_url": "http://localhost:11434/v1",
         "analysis_model": "qwen2.5:14b", "live_model": "qwen2.5:7b"}}
```

`analysis_model` is used for claim extraction and briefs, and
`live_model` for the live suggestions, which should be fast. `research analyze` shows a
token estimate and asks before spending. Optional `embedding_model`
enables semantic search on top of the built-in full-text search.

**Latency.** Debate mode transcribes in 8-second blocks
(`debate.chunk_duration_s`), and a suggestion adds one fast-model call
(typically 1–4s). If the model falls behind, stale points are skipped so
suggestions stay current. Every suggestion is logged to
`~/.kickvoice/debates/*.jsonl` for review afterwards.

### 8. Agent mode (MCP): a live chat copilot

`kickvoice mcp` runs kickvoice as an [MCP](https://modelcontextprotocol.io)
server, so an AI agent (Claude Desktop, Claude Code, or any MCP client) can
run the whole loop for you:
- follow the stream,
- work out who's speaking (and enroll a new voice),
- research the opponent in the background,
- draft chat replies to their arguments, grounded in their record.

You pick a reply and paste it into chat yourself. The agent can copy your
chosen reply to the clipboard, but **never posts to chat**.

**Connect it:**

```bash
# Claude Code
claude mcp add kickvoice -- /path/to/kickvoice/run.sh mcp
```

```json
// Claude Desktop: claude_desktop_config.json
{"mcpServers": {"kickvoice": {"command": "/path/to/kickvoice/run.sh", "args": ["mcp"]}}}
```

Then ask: *"Use the live_chat_copilot prompt for https://kick.com/somechannel,
opponent Jane Pundit."* Or just describe what you want; the server's
instructions explain the workflow to the agent.

**How the loop works:**

```
start_live_session ─▶ get_live_updates (long-poll, cursor) ─▶ "line" / "point" / "voices" events
                                                                    │
        unknown voice? ─▶ list_voices ─▶ (you confirm) enroll_voice ─▶ set_opponent
                                                                    │ point
        get_point_evidence ─▶ agent drafts replies ─▶ verify_message ─▶ you pick ─▶ copy_to_clipboard
        (or suggest_counters: replies from kickvoice's own model, already verified)

   thin record? research_find_identity ─▶ (you confirm) research_link_wikidata
                ─▶ research_collect / research_analyze (background jobs, job_status)
```

| Tools | What for |
|---|---|
| `start_live_session`, `get_live_updates`, `stop_live_session`, `list_live_sessions` | Follow a stream; events carry transcript lines and the opponent's points |
| `list_voices`, `enroll_voice`, `set_opponent` | Who's speaking; turn an unknown voice into a profile mid-stream |
| `research_subjects`, `research_profile`, `research_search` | What's on record |
| `research_find_identity`, `research_link_wikidata`, `research_add_subject` | Identify someone (people only; links nothing without a confirm step) |
| `research_collect`, `research_analyze`, `job_status` | Background research jobs |
| `get_point_evidence`, `suggest_counters` | Evidence to write a reply / ready-made replies |
| `verify_message`, `format_for_chat`, `copy_to_clipboard` | Check a reply, make it chat-shaped, copy it |

**Keeping replies accurate.** A chat message can't be corrected once it's
sent, so `verify_message` checks every reply before it's offered:
- **Quotes must be exact.** Anything in quote marks must appear *exactly*, as
  one unbroken passage, in the opponent's own first-hand statements. Only case
  and punctuation may differ. One changed word ("raise" → "abolish") fails.
  - **No skipped words.** `...` can't stand in for words that were left out,
    so "are … fine" is not a quote of "are not fine", and two sentences can't
    be joined into one. A quote may still trail off or pick up mid-sentence
    ("…housing supply has to grow"), because nothing inside it is missing.
  - **Any quote marks count:** `"…"`, `“…”`, `'…'`, `‘…’`, `«…»`, `„…“`.
    Single marks are also apostrophes, so "don't" and "Jane's" are not quotes,
    but when it's unclear where a single-quoted passage ends (a plural
    possessive such as "the streamers' rights" looks like a closing mark), the
    longest reading is checked. That can reject a reply that was fine; the
    message says to use double quote marks, which are never ambiguous.
  - Quoted spans shorter than 8 characters (a single word in scare quotes) are
    not treated as quotations. When the opponent has no research record,
    replies with anything in quote marks are rejected, since nothing can be
    checked.
- **Third-party words can't be passed off as theirs.** Words only found in
  reports *about* them fail as quotes.
- **Years need dated evidence.** A year in the reply must match evidence
  from that year.
- **"You said…" needs a source.** Claims like that must cite something they
  actually said.
- **It has to fit chat.** Replies are single-line and at most
  `agent.chat_max_chars` characters (default 300). Links are flagged,
  because many chats block them.

`suggest_counters` runs these checks on its own replies. An agent writing its
own replies is told to run `verify_message` first.

**Needs:** `pip install -e ".[agent]"` (setup installs it). Server-side
suggestions and analysis use your configured model (`ANTHROPIC_API_KEY`
etc.). Without a key, the agent can still use `get_point_evidence` and
write replies itself. Logs go to stderr, since stdout carries the protocol.

## Make Targets

```
make help              Show all targets
make setup             Full interactive setup (./setup.sh)
make install           Non-interactive install into .venv (CPU torch)
make install-gpu       Non-interactive install into .venv (CUDA torch)
make download-models   Pre-download ML models
make verify            Check all imports work
make test              Run the test suite
make coverage          Tests + coverage report (fails under 90%)
make lint / format     ruff check / auto-fix + format
make build             Build sdist + wheel into dist/
make check-dist        Build and validate with twine
make clean / reset     Remove build & cache artifacts / the venv
```

## Architecture

```
                    Pipeline 1: Enrollment (offline)
┌─────────┐    ┌────────┐    ┌─────┐    ┌───────────┐    ┌──────────┐
│ YouTube  │───▶│ ffmpeg │───▶│ VAD │───▶│ ECAPA-TDNN│───▶│ Profile  │
│ / File   │    │normalize│    │     │    │ (embed)   │    │ Store    │
└─────────┘    └────────┘    └─────┘    └───────────┘    └──────────┘

                    Pipeline 2: Live Monitoring
┌──────────┐    ┌───────────┐    ┌─────┐    ┌───────────┐    ┌──────────┐
│ Kick.com │───▶│streamlink │───▶│ VAD │───▶│ ECAPA-TDNN│───▶│ Cosine   │
│ HLS      │    │ + ffmpeg  │    │     │    │ (embed)   │    │ Similarity│
└──────────┘    │ (10s chunks)   └─────┘    └───────────┘    │ vs profiles│
                └───────────┘                                 └──────────┘
                                                                   │
                                                              ┌────▼─────┐
                                                              │ Speaker  │
                                                              │ Identified│
                                                              └──────────┘

                    Pipeline 3: Profile Extraction
┌──────────┐   ┌────────────┐   ┌─────┐   ┌────────────┐   ┌─────────────┐
│ Stream / │──▶│ streamlink │──▶│ VAD │──▶│ ECAPA on 2s│──▶│ Agglomerative│
│ File     │   │ + ffmpeg   │   │     │   │ windows    │   │ clustering   │
└──────────┘   │ (30s blocks)   └─────┘   └────────────┘   └──────┬──────┘
               └────────────┘                                     │ one group per voice
               ┌──────────────┐   ┌───────────────┐   ┌───────────▼──────────┐
               │ Profile Store│◀──│ User approval │◀──│ Match vs profiles +  │
               │ (new/merged) │   │ (review clip) │   │ save session         │
               └──────────────┘   └───────────────┘   └──────────────────────┘
```

```
                    Pipeline 4: Speaker-attributed transcription
┌──────────┐  ┌─────┐  ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐
│ Stream / │─▶│ VAD │─▶│ Whisper      │─▶│ each word → voice│─▶│ lines (turns)│
│ File     │  │     │  │ (word times) │  │ windows vs profiles │ provisional  │
└──────────┘  └─────┘  └──────────────┘  └──────────────────┘  └──────┬───────┘
                                                                      │ end of session
        ┌──────────────┐   ┌──────────────────────────────────────────▼───────┐
        │ transcripts  │◀──│ group lines by voice · 1 voice group per profile │
        │ .db (SQLite) │   │ · confident only if group AND line match         │
        └──────────────┘   └──────────────────────────────────────────────────┘
```

Extraction cuts speech into short overlapping windows (2s, 1s hop) so that a
single embedding rarely spans two people taking turns. It then groups them
with average-linkage clustering on cosine distance, so the number of speakers
doesn't need to be known in advance. Voices with too little total speech are
dropped as noise.

### Key Components

| Component | What it does |
|---|---|
| **Silero VAD** | Strips silence, music, and game audio — keeps only speech |
| **ECAPA-TDNN** | Extracts 192-dim speaker embeddings (voice identity vectors) |
| **Cosine Similarity** | Compares live embeddings to enrolled profiles |
| **streamlink** | Resolves Kick.com HLS stream URLs |
| **ffmpeg** | Audio format conversion, normalization, PCM extraction |

### Why not Dejavu?

Dejavu is an **audio fingerprinting** tool (like Shazam). It matches *exact recordings*,
not voices. If you fingerprint someone saying "hello," Dejavu only matches that
exact "hello" — not the same person saying "goodbye."

This project uses **speaker embeddings** (ECAPA-TDNN), which encode *who* is speaking
regardless of *what* they say.

## Configuration

Config is stored at `~/.kickvoice/config.json`. Override via CLI:

```bash
# Lower threshold for more matches (more false positives)
kickvoice monitor --stream "https://kick.com/channel" --threshold 0.50

# Use verbose logging to debug
kickvoice -v monitor --stream "https://kick.com/channel"
```

### Tuning the threshold

| Threshold | Behavior |
|---|---|
| 0.50 | Aggressive — more matches, more false positives |
| 0.60 | Default — balanced |
| 0.70 | Conservative — fewer matches, fewer false positives |
| 0.75+ | High confidence only |

### Tuning profile extraction

Settings live in the `extract` section of `~/.kickvoice/config.json`. The two
that matter most are also CLI flags:

| Setting | Flag | Default | Effect |
|---|---|---|---|
| `cluster_distance_threshold` | `--cluster-threshold` | 0.70 | Lower → splits voices more readily (one person may become two groups). Higher → merges more (two similar voices may become one group). |
| `min_speaker_speech_s` | `--min-speech` | 20 | Voices with less total speech are discarded. |
| `capture_duration_s` | `--duration` | 300 | Listening time for streams (0 = until Ctrl+C / stream end). |
| `window_s` / `window_hop_s` | — | 2.0 / 1.0 | Embedding window size / step. |
| `sample_clip_s` | — | 20 | Length of each review clip. |

The right clustering threshold depends on the embedding model. 0.70 is
chosen for ECAPA's score range. If a review clip contains two different
people, re-run with a lower `--cluster-threshold`. If the same person keeps
appearing as several voices, raise it.

### Tuning transcription

Settings live in the `transcribe` section of `config.json`:

| Setting | Flag | Default | Effect |
|---|---|---|---|
| `asr_model` | `--model` | small | Whisper size: tiny/base/small/medium/large-v3, or a local path. |
| `language` | `--language` | auto | Fixing it (e.g. `en`) avoids mis-detection on short clips. |
| `asr_device` | `--device` | cpu | `cuda` for GPU. |
| `keep_unattributed` | `--drop-unknown` | true | Keep unrecognised voices' text as context. |
| `min_confident_turn_s` | — | 2.0 | Shorter lines can't be "confident". |
| `max_pause_s` | — | 1.0 | A pause this long starts a new line. |

Voice matching uses `identify.match_threshold` / `high_confidence_threshold`
(same as `monitor`). Grouping voices uses `extract.cluster_distance_threshold`
(same as `extract-profiles`), so calibrating that once covers both.

**Known limitation:** two similar-sounding people who talk over each other
a lot (for example, rapid back-and-forth co-hosts on the same mic mix) can
end up in one group. Always listen to the review clip before accepting.

## Project Structure

```
kickvoice/
├── pyproject.toml        # Package metadata, dependencies/extras, pytest/coverage/ruff config
├── src/kickvoice/
│   ├── cli.py            # Click CLI — thin commands that call a pipeline or spawn an agent
│   ├── cli_research.py   # `research …` and `debate` commands
│   ├── config.py         # Configuration dataclasses (defaults + JSON file)
│   ├── console.py        # Output formatting
│   ├── audio/            # ffmpeg, stream/file capture, VAD, yt-dlp downloads
│   ├── speakers/         # ECAPA embeddings, profiles, identification, clustering, packs
│   ├── transcripts/      # Whisper ASR, word→speaker attribution, session refinement, store
│   ├── research/         # Polite HTTP, LLM clients, sources, claims, retrieval, counters, briefs
│   ├── pipelines/        # enroll, monitor, extract, review, transcribe, research, debate
│   └── agent/            # The agentic layer
│       ├── server.py     #   MCP server (`kickvoice mcp`): tools + prompt
│       ├── service.py    #   tool logic as plain Python
│       ├── live.py       #   background live sessions + pollable event log
│       ├── jobs.py       #   background research jobs
│       ├── chat.py       #   chat formatting, reply fact-checking, clipboard
│       ├── runner.py     #   spawns a goal agent for a CLI command; writes outputs
│       ├── goals/        #   extract-profiles and monitor agents (llm + rules policies)
│       ├── episode.py    #   bounded model-and-tools loop, token budget
│       ├── toolbox.py    #   MCP client connection + local tools, all traced
│       ├── toolcalling.py#   Anthropic / OpenAI-compatible tool calling
│       ├── terminal.py   #   the person at the keyboard (questions + commands)
│       └── trace.py      #   run folders and trace.jsonl
├── tests/                # ~390 tests, no models or network needed
├── setup.sh / run.sh     # Interactive venv setup / run inside it
└── Makefile
```

## Development

```bash
pip install -e ".[all,dev]"      # or: make install
make lint                        # ruff (rules + formatting are enforced in CI)
make coverage                    # pytest with branch coverage, fails under 90%
make check-dist                  # build the sdist/wheel and validate them
```

- **No models or network needed.** Synthetic voices (pitched tones) stand in
  for speakers, with fake encoder, ASR and model components. Fake HTTP
  servers stand in for Wikipedia, X, Reddit and the model APIs. Fake
  `streamlink`/`yt-dlp` executables stand in for the network tools.
- **The real pipeline code runs end to end.** That includes the agents
  spawning a real MCP server subprocess and talking to it over stdio.
- **Subprocess coverage counts.** Coverage includes the code running inside
  the spawned MCP servers (`[tool.coverage.run] patch = ["subprocess"]`).
- **Slow tests are marked `slow`.** `pytest -m "not slow"` skips those that
  launch whole CLI processes.
- See `CONTRIBUTING.md` for the workflow and `CHANGELOG.md` for releases.

### Code layout in one sentence

`audio/` gets sound into one shape (16kHz mono PCM), `speakers/` decides whose
voice it is, `transcripts/` turns speech into attributed text, `research/`
builds the cited record, `pipelines/` are the end-to-end flows, `agent/`
exposes all of it over MCP and runs goal-driven agents on top, and `cli.py`
only parses options and prints results.

### CLI notes

The CLI is built with [Click](https://click.palletsprojects.com/): `-h/--help`
works on every command and subcommand, and names are positional:

```bash
./run.sh enroll "xQc" --file clip.wav
./run.sh profiles info "xQc"
./run.sh profiles import streamers.json
```

The older flag forms (`enroll --name "xQc"`, `profiles info --name "xQc"`,
`profiles import --file pack.json`, comma-separated `--youtube a,b`) still
work, so existing scripts don't break.

## Requirements

- Python 3.10–3.12, ffmpeg
- ~1GB disk for CPU PyTorch + models (several GB with CUDA PyTorch); Whisper
  `small` adds ~500MB
- ~2–4GB RAM during live transcription
- Agents with `--policy llm`: an LLM API key or a local OpenAI-compatible server
