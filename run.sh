#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────
# run.sh — runs kickvoice inside the venv without manual activation
#
# Usage:
#   ./run.sh enroll "Speaker" --file audio.wav
#   ./run.sh monitor --stream "https://kick.com/channel"
#   ./run.sh extract-profiles --stream "https://kick.com/channel"
#   ./run.sh profiles list
# ──────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Virtual environment not found at ${VENV_DIR}"
    echo "   Run setup first: ./setup.sh"
    exit 1
fi

# Put the venv's bin/ on PATH so the CLI tools installed there (streamlink,
# yt-dlp) are found by subprocess calls, just like an activated venv.
export PATH="${VENV_DIR}/bin:${PATH}"

exec "$VENV_PYTHON" -m kickvoice.cli "$@"
