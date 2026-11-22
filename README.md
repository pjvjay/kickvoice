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
./run.sh enroll --name "DrDisrespect" --file /path/to/audio.wav
./run.sh enroll --name "xQc" --file /path/to/clip.mp3

# Option B: activate venv manually
source .venv/bin/activate
kickvoice enroll --name "DrDisrespect" --file /path/to/audio.wav
```

From YouTube videos (requires `yt-dlp`, installed automatically with setup):

```bash
./run.sh enroll --name "DrDisrespect" \
  --youtube "https://youtube.com/watch?v=ABC,https://youtube.com/watch?v=DEF"
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
./run.sh profiles info --name "DrDisrespect"
./run.sh profiles remove --name "xQc"

# Profile packs: portable JSON bundles of voiceprints
./run.sh profiles export --out streamers.json                  # all profiles
./run.sh profiles export --out two.json --name "xQc,Adin Ross"  # selected
./run.sh profiles import --file streamers.json                 # skip existing names
./run.sh profiles import --file streamers.json --on-conflict merge   # or: overwrite
```

A pack is a single diff-friendly JSON file (~3 KB per profile), which makes it
easy to keep a shared library of known streamer voices in a git repo and grow
it with `extract-profiles`. I'm not aware of any existing public repository
of streamer voiceprints, so packs are the way to build one.

Packs record the embedding model that produced them. Voiceprints from
different models aren't comparable, so importing a pack made with a
different model is refused.

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
│   ├── cli.py            # CLI entrypoint (enroll / monitor / extract-profiles / review / profiles)
│   ├── config.py         # Configuration dataclasses
│   ├── enroll/
│   │   └── enroll.py     # Speaker enrollment from files/YouTube
│   ├── extract/
│   │   ├── cluster.py    # Windowing + agglomerative speaker clustering
│   │   ├── extractor.py  # Stream → voices → candidates pipeline
│   │   ├── review.py     # Interactive approval (create / merge / skip)
│   │   └── session.py    # On-disk extraction sessions + review clips
│   ├── live/
│   │   ├── capture.py    # Stream/file audio capture (streamlink + ffmpeg)
│   │   ├── detect.py     # VAD (Silero + energy fallback)
│   │   ├── identify.py   # Cosine similarity matching
│   │   └── monitor.py    # Main pipeline loop
│   ├── models/
│   │   └── ecapa.py      # ECAPA-TDNN speaker encoder wrapper
│   └── storage/
│       ├── profiles.py   # Speaker profile CRUD (JSON index + .npy)
│       └── packs.py      # Profile pack export/import
├── tests/                # pytest suite (no models/network needed)
└── .venv/                # Created by setup.sh (gitignored)
```

## Development

```bash
make install              # Sets up venv + deps
source .venv/bin/activate # Activate for interactive dev
pytest tests/             # Run tests
make clean                # Remove pycache
```

## Requirements

- Python 3.10+
- ffmpeg
- ~200MB disk for CPU PyTorch + models (~2GB if using GPU PyTorch)
- ~2GB RAM during monitoring
