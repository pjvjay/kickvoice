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

### 5. Manage profiles

```bash
./run.sh profiles list
./run.sh profiles info --name "DrDisrespect"
./run.sh profiles remove --name "xQc"
```

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
```

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
│   ├── cli.py            # CLI entrypoint (enroll / monitor / profiles)
│   ├── config.py         # Configuration dataclasses
│   ├── enroll/
│   │   └── enroll.py     # Speaker enrollment from files/YouTube
│   ├── live/
│   │   ├── capture.py    # Stream/file audio capture (streamlink + ffmpeg)
│   │   ├── detect.py     # VAD (Silero + energy fallback)
│   │   ├── identify.py   # Cosine similarity matching
│   │   └── monitor.py    # Main pipeline loop
│   ├── models/
│   │   └── ecapa.py      # ECAPA-TDNN speaker encoder wrapper
│   └── storage/
│       └── profiles.py   # Speaker profile CRUD (JSON index + .npy)
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
