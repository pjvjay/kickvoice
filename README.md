# kickvoice

**Speaker identification on Kick.com live streams.**

Detects *who* is speaking on a live stream by comparing voice embeddings against
enrolled speaker profiles. Uses SpeechBrain's ECAPA-TDNN model (trained on VoxCeleb)
for speaker-discriminative embeddings.

> ⚠️ **Not Dejavu.** This is speaker *recognition* (identify a person by voice),
> not audio *fingerprinting* (match a specific recording). See [Architecture](#architecture).

## Prerequisites

- Python 3.10+
- ffmpeg (`brew install ffmpeg` / `sudo apt install ffmpeg`)

## Quick Start

### 1. Setup (one time)

The interactive setup script creates a virtual environment, installs all
dependencies (PyTorch, SpeechBrain, streamlink, etc.), and optionally
pre-downloads the ML models:

```bash
git clone <this-repo>
cd kickvoice
chmod +x setup.sh run.sh
./setup.sh
```

The setup will prompt you to choose CPU or GPU PyTorch and whether to
download models now.

**Alternative: non-interactive setup via Make**

```bash
make install            # CPU PyTorch (smaller, ~200MB)
make install-gpu        # CUDA GPU PyTorch (~2GB)
make download-models    # Pre-download ML models
```

### 2. Enroll speakers

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

### 3. Monitor a Kick.com stream

```bash
./run.sh monitor --stream "https://kick.com/somechannel"
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

### 4. Test with a local file

```bash
./run.sh monitor --file recording.wav
```

### 5. Extract profiles from a stream (discover who's talking)

Don't have clean enrollment audio? Let kickvoice listen to a stream, work out
how many distinct voices there are, and build profiles from them — with you
approving every one.

```bash
./run.sh extract-profiles --stream "https://kick.com/somechannel"
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

### 6. Manage and share profiles

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

### 7. Transcribe: who said what

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

### 8. Research an opponent, then get live counter-arguments

Before a debate, kickvoice builds a cited record of what your opponent has
said publicly. During the debate, it listens for their points and suggests
replies grounded in that record.

```bash
# 1. Who are they? Link accounts (yours are trusted; discovered ones need your OK)
./run.sh research add "Jane Pundit" --x JanePundit --youtube @janepundit --voice Jane
./run.sh research discover "Jane Pundit" --x JanePundit     # official accounts via Wikidata

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

## Make Targets

```
make help              Show all targets
make setup             Full interactive setup
make install           Non-interactive install (CPU torch)
make install-gpu       Non-interactive install (CUDA torch)
make download-models   Pre-download ML models
make verify            Check all imports work
make profiles          List enrolled speakers
make test              Run tests
make clean             Remove pycache/build artifacts
make reset             Remove venv entirely
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
├── setup.sh              # Interactive setup — creates venv + installs everything
├── run.sh                # Wrapper — runs kickvoice without manual venv activation
├── Makefile              # Make targets for setup, install, test, clean
├── pyproject.toml        # Package metadata and dependencies
├── kickvoice/
│   ├── __init__.py
│   ├── __main__.py       # Enables `python -m kickvoice`
│   ├── cli.py            # Click CLI — thin commands that call a pipeline
│   ├── cli_research.py   # `research …` and `debate` commands
│   ├── config.py         # Configuration dataclasses (defaults + JSON file)
│   ├── console.py        # Output formatting (messages, tables, durations)
│   ├── audio/            # Getting audio in, in one shape: 16kHz mono PCM
│   │   ├── ffmpeg.py     # Decoding, PCM conversions, wav writing
│   │   ├── capture.py    # Stream (streamlink) + file capture, chunking
│   │   ├── vad.py        # Voice activity detection (Silero + energy fallback)
│   │   └── download.py   # yt-dlp downloads
│   ├── speakers/         # What a voice *is*
│   │   ├── encoder.py    # ECAPA-TDNN embeddings
│   │   ├── profiles.py   # Profile storage (JSON index + .npy voiceprints)
│   │   ├── identify.py   # Cosine similarity matching
│   │   ├── cluster.py    # Windowing + agglomerative speaker clustering
│   │   └── packs.py      # Profile pack export/import
│   ├── transcripts/      # Who said what
│   │   ├── asr.py        # Speech-to-text interface + faster-whisper
│   │   ├── attribution.py# Word → speaker alignment rules (pure functions)
│   │   ├── refine.py     # Whole-session attribution (voice groups)
│   │   └── store.py      # SQLite transcript store + full-text search
│   ├── research/         # Opposition research + live counters
│   │   ├── net.py        # Polite HTTP (rate limits, cache, robots.txt)
│   │   ├── llm.py        # Anthropic / OpenAI-compatible clients, JSON validation
│   │   ├── store.py      # SQLite: people, accounts, documents, claims (+ search)
│   │   ├── sources/      # Wikipedia/Wikidata, YouTube, X, Reddit, web, local
│   │   ├── extract.py    # Claim extraction with verbatim-quote verification
│   │   ├── retrieve.py   # Hybrid retrieval (full-text + optional embeddings)
│   │   ├── counter.py    # Live counter-argument engine (validated citations)
│   │   └── brief.py      # Pre-debate brief
│   └── pipelines/        # End-to-end flows
│       ├── enroll.py     # Sources → profile
│       ├── monitor.py    # Stream → who is speaking now
│       ├── extract.py    # Stream → distinct voices → candidates
│       ├── review.py     # Interactive approval (create / merge / skip)
│       ├── session.py    # On-disk extraction sessions + review clips
│       ├── transcribe.py # Stream → speaker-attributed transcript
│       ├── research.py   # Discover accounts, collect public record
│       └── debate.py     # Live: opponent's points → counter-arguments
├── tests/                # pytest suite (no models/network needed)
└── .venv/                # Created by setup.sh (gitignored)
```

## Development

```bash
make install              # Sets up venv + deps
source .venv/bin/activate # Activate for interactive dev
pytest tests/             # Run tests (no models or network needed)
make clean                # Remove pycache
```

The test suite runs without ML models or network access: synthetic tones stand
in for voices, and the encoder/VAD/capture components are injected, so the
pipelines can be exercised end to end in seconds.

### Code layout in one sentence

`audio/` gets sound into one shape (16kHz mono PCM), `speakers/` decides whose
voice it is, `transcripts/` turns speech into attributed text, `pipelines/`
are the end-to-end flows, and `cli.py` only parses options and prints results.

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

- Python 3.10+
- ffmpeg
- ~200MB disk for CPU PyTorch + models (~2GB if using GPU PyTorch)
- ~2GB RAM during monitoring
