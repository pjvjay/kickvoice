#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────
# kickvoice setup — creates venv, installs deps, verifies tools
# ──────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
MIN_PYTHON="3.10"

# Detect platform early — needed for Python selection
OS="$(uname -s)"
ARCH="$(uname -m)"
IS_INTEL_MAC=false
if [ "$OS" = "Darwin" ] && [ "$ARCH" = "x86_64" ]; then
    IS_INTEL_MAC=true
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { printf "${CYAN}i ${NC} %s\n" "$*" >&2; }
ok()    { printf "${GREEN}OK${NC} %s\n" "$*" >&2; }
warn()  { printf "${YELLOW}!! ${NC} %s\n" "$*" >&2; }
fail()  { printf "${RED}ERR${NC} %s\n" "$*" >&2; exit 1; }

# ── Helper: find a usable Python ─────────────────────────────
find_python() {
    # On Intel Mac, we MUST use Python 3.12 (last version with torch x86_64 wheels)
    if [ "$IS_INTEL_MAC" = true ]; then
        warn "Intel Mac detected — PyTorch requires Python 3.12 (no x86_64 wheels for 3.13+)"

        # Check known locations for python3.12 (brew installs here on Intel Macs)
        local CANDIDATES="/usr/local/bin/python3.12 /opt/homebrew/bin/python3.12"

        for candidate in $CANDIDATES; do
            if [ -x "$candidate" ]; then
                local ver
                ver="$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "unknown")"
                if [ "$ver" = "3.12" ]; then
                    echo "$candidate"
                    return 0
                fi
            fi
        done

        # Not found — print instructions and exit
        fail "Python 3.12 not found. Install it with: brew install python@3.12"
    fi

    # Non-Intel-Mac: use PYTHON env var or default python3
    local py="${PYTHON:-python3}"
    if [ -x "$(which "$py" 2>/dev/null || true)" ]; then
        echo "$py"
        return 0
    fi
    fail "Python not found. Install Python ${MIN_PYTHON}+ and retry."
}

# ── 1. Find and verify Python ────────────────────────────────
info "Checking Python..."
PYTHON="$(find_python)"

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    fail "Python ${MIN_PYTHON}+ required (found ${PY_VERSION})"
fi
ok "Python ${PY_VERSION} (${PYTHON})"

# ── 2. Check system dependencies ─────────────────────────────
info "Checking system dependencies..."

if ! command -v ffmpeg &>/dev/null; then
    warn "ffmpeg not found. Install it:"
    echo "    macOS:  brew install ffmpeg"
    echo "    Ubuntu: sudo apt install ffmpeg"
    echo "    Arch:   sudo pacman -S ffmpeg"
    fail "ffmpeg is required"
fi
ok "ffmpeg $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"

# ── 3. Create virtual environment ────────────────────────────
if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment already exists at ${VENV_DIR}"
    read -rp "    Recreate it? (y/N): " RECREATE
    if [[ "$RECREATE" =~ ^[Yy]$ ]]; then
        info "Removing old venv..."
        rm -rf "$VENV_DIR"
    else
        info "Reusing existing venv"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment at ${VENV_DIR}..."
    "$PYTHON" -m venv "$VENV_DIR"
    ok "Virtual environment created"
fi

# ── 4. Activate and upgrade pip ──────────────────────────────
info "Activating venv and upgrading pip..."
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel --quiet
ok "pip $(pip --version | awk '{print $2}')"

# ── 5. Install PyTorch ────────────────────────────────────────
info "Installing PyTorch..."

if [ "$IS_INTEL_MAC" = true ]; then
    # Intel Mac — we already forced Python 3.12, now pin torch 2.2.2
    info "Installing PyTorch 2.2.2 (last Intel Mac release)..."
    pip install "torch==2.2.2" "torchaudio==2.2.2" --quiet
    ok "PyTorch 2.2.2 installed (Intel Mac)"
elif [ "$OS" = "Darwin" ]; then
    # Apple Silicon — latest PyTorch from PyPI works natively
    info "Detected macOS Apple Silicon — installing latest PyTorch..."
    pip install torch torchaudio --quiet
    ok "PyTorch installed (macOS Apple Silicon)"
else
    # Linux — offer CPU vs CUDA choice
    echo ""
    echo "    Select PyTorch variant:"
    echo "    [1] CPU only (smaller, ~200MB) — recommended for most users"
    echo "    [2] CUDA 12.x (GPU, ~2GB) — for NVIDIA GPU acceleration"
    echo "    [3] Skip (already installed)"
    echo ""
    read -rp "    Choice [1]: " TORCH_CHOICE
    TORCH_CHOICE="${TORCH_CHOICE:-1}"

    case "$TORCH_CHOICE" in
        1)
            info "Installing PyTorch CPU..."
            pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
            ;;
        2)
            info "Installing PyTorch CUDA 12.x..."
            pip install torch torchaudio --quiet
            ;;
        3)
            info "Skipping PyTorch install"
            ;;
        *)
            warn "Invalid choice, defaulting to CPU"
            pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
            ;;
    esac
    ok "PyTorch installed"
fi

# ── 6. Install kickvoice and remaining deps ──────────────────
info "Installing kickvoice and dependencies..."
pip install -e "${SCRIPT_DIR}[youtube,transcribe,agent,dev]" --quiet
ok "kickvoice installed"

# ── 7. Install streamlink ────────────────────────────────────
info "Installing streamlink..."
pip install streamlink --quiet
ok "streamlink $(streamlink --version 2>&1 | head -1)"

# ── 8. Verify installation ───────────────────────────────────
echo ""
info "Verifying installation..."
VENV_PYTHON="${VENV_DIR}/bin/python"
"$VENV_PYTHON" -c "
import sys
errors = []

try:
    import torch
    print(f'  torch        {torch.__version__}')
except ImportError as e:
    errors.append(f'torch: {e}')

try:
    import torchaudio
    print(f'  torchaudio   {torchaudio.__version__}')
except ImportError as e:
    errors.append(f'torchaudio: {e}')

try:
    import speechbrain
    print(f'  speechbrain  {speechbrain.__version__}')
except ImportError as e:
    errors.append(f'speechbrain: {e}')

try:
    import streamlink
    print(f'  streamlink   {streamlink.__version__}')
except ImportError as e:
    errors.append(f'streamlink: {e}')

try:
    import click
    print(f'  click        {click.__version__}')
except ImportError as e:
    errors.append(f'click: {e}')

try:
    import faster_whisper
    print(f'  faster-whisper {faster_whisper.__version__}')
except ImportError as e:
    errors.append(f'faster-whisper: {e}')

try:
    import numpy
    print(f'  numpy        {numpy.__version__}')
except ImportError as e:
    errors.append(f'numpy: {e}')

try:
    import kickvoice
    print(f'  kickvoice    {kickvoice.__version__}')
except ImportError as e:
    errors.append(f'kickvoice: {e}')

if errors:
    print()
    for err in errors:
        print(f'  ⚠  MISSING: {err}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    ok "All dependencies verified!"
else
    echo ""
    fail "Some dependencies are missing — see above"
fi

# ── 9. Pre-download models (optional) ────────────────────────
echo ""
read -rp "Download speaker recognition models now? (~80MB) (Y/n): " DL_MODELS
DL_MODELS="${DL_MODELS:-Y}"
if [[ "$DL_MODELS" =~ ^[Yy]$ ]]; then
    info "Downloading ECAPA-TDNN model (this may take a minute)..."
    "$VENV_PYTHON" -c "
from speechbrain.inference.speaker import EncoderClassifier
import os
savedir = os.path.expanduser('~/.kickvoice/pretrained_models/ecapa')
model = EncoderClassifier.from_hparams(
    source='speechbrain/spkrec-ecapa-voxceleb',
    savedir=savedir,
    run_opts={'device': 'cpu'},
)
print('Model downloaded and cached')
"
    ok "ECAPA-TDNN model ready"

    info "Loading Silero VAD model..."
    "$VENV_PYTHON" -c "
from silero_vad import load_silero_vad
load_silero_vad()
print('Silero VAD ready')
"
    ok "Silero VAD model ready"
fi

# ── Done ──────────────────────────────────────────────────────
echo ""
echo "========================================================"
printf "  ${GREEN}Setup complete!${NC}\n"
echo "========================================================"
echo ""
echo "  Activate the venv in your shell:"
printf "    ${CYAN}source ${VENV_DIR}/bin/activate${NC}\n"
echo ""
echo "  Then use kickvoice:"
printf "    ${CYAN}kickvoice enroll \"Speaker\" --file audio.wav${NC}\n"
printf "    ${CYAN}kickvoice extract-profiles --stream \"https://kick.com/channel\"${NC}\n"
printf "    ${CYAN}kickvoice monitor --stream \"https://kick.com/channel\"${NC}\n"
printf "    ${CYAN}kickvoice profiles list${NC}\n"
echo ""
echo "  Or use the wrapper script (auto-activates venv):"
printf "    ${CYAN}./run.sh enroll \"Speaker\" --file audio.wav${NC}\n"
printf "    ${CYAN}./run.sh extract-profiles --stream \"https://kick.com/channel\"${NC}\n"
printf "    ${CYAN}./run.sh monitor --stream \"https://kick.com/channel\"${NC}\n"
printf "    ${CYAN}./run.sh transcribe --stream \"https://kick.com/channel\"${NC}\n"
echo ""
