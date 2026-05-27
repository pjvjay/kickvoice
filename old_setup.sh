#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────
# kickvoice setup — creates venv, installs deps, verifies tools
# ──────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON="${PYTHON:-python3}"
MIN_PYTHON="3.10"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}ℹ ${NC} $*"; }
ok()    { echo -e "${GREEN}✅${NC} $*"; }
warn()  { echo -e "${YELLOW}⚠ ${NC} $*"; }
fail()  { echo -e "${RED}❌${NC} $*"; exit 1; }

# ── 1. Check Python version ──────────────────────────────────
info "Checking Python..."
if ! command -v "$PYTHON" &>/dev/null; then
    fail "Python not found. Install Python ${MIN_PYTHON}+ and retry."
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    fail "Python ${MIN_PYTHON}+ required (found ${PY_VERSION})"
fi
ok "Python ${PY_VERSION}"

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

# ── 5. Install PyTorch (CPU by default, GPU optional) ────────
info "Installing PyTorch..."
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

# ── 6. Install kickvoice and remaining deps ──────────────────
info "Installing kickvoice and dependencies..."
pip install -e "${SCRIPT_DIR}[youtube,dev]" --quiet
ok "kickvoice installed"

# ── 7. Install streamlink ────────────────────────────────────
info "Installing streamlink..."
pip install streamlink --quiet
ok "streamlink $(streamlink --version 2>&1 | head -1)"

# ── 8. Verify installation ───────────────────────────────────
echo ""
info "Verifying installation..."
"$PYTHON" -c "
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
    "$PYTHON" -c "
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

    info "Downloading Silero VAD model..."
    "$PYTHON" -c "
import torch
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    onnx=False,
)
print('Silero VAD downloaded and cached')
"
    ok "Silero VAD model ready"
fi

# ── Done ──────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}  Setup complete!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Activate the venv in your shell:"
echo -e "    ${CYAN}source ${VENV_DIR}/bin/activate${NC}"
echo ""
echo "  Then use kickvoice:"
echo -e "    ${CYAN}kickvoice enroll --name \"Speaker\" --file audio.wav${NC}"
echo -e "    ${CYAN}kickvoice monitor --stream \"https://kick.com/channel\"${NC}"
echo -e "    ${CYAN}kickvoice profiles list${NC}"
echo ""
echo "  Or use the wrapper script (auto-activates venv):"
echo -e "    ${CYAN}./run.sh enroll --name \"Speaker\" --file audio.wav${NC}"
echo -e "    ${CYAN}./run.sh monitor --stream \"https://kick.com/channel\"${NC}"
echo ""
