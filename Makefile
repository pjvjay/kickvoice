.PHONY: setup clean reset test enroll-test monitor-test profiles help

VENV      := .venv
PYTHON    := $(VENV)/bin/python
PIP       := $(VENV)/bin/pip
KICKVOICE := $(VENV)/bin/kickvoice

# Detect Intel Mac — needs python3.12 for torch compatibility
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S)-$(UNAME_M),Darwin-x86_64)
  SYSTEM_PYTHON := python3.12
  IS_INTEL_MAC := true
else
  SYSTEM_PYTHON := python3
  IS_INTEL_MAC := false
endif

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## Run full setup (venv + deps + models)
	@chmod +x setup.sh && ./setup.sh

venv: ## Create venv only (no interactive prompts)
	@test -d $(VENV) || $(SYSTEM_PYTHON) -m venv $(VENV)
	@$(PIP) install --upgrade pip setuptools wheel --quiet
	@echo "✅ venv ready at $(VENV) ($(SYSTEM_PYTHON))"

install: venv ## Install kickvoice into venv (auto-detects platform)
	@if [ "$(IS_INTEL_MAC)" = "true" ]; then \
		echo "⚠  Intel Mac — installing PyTorch 2.2.2 with python3.12"; \
		$(PIP) install "torch==2.2.2" "torchaudio==2.2.2" --quiet; \
	elif [ "$$(uname -s)" = "Darwin" ]; then \
		$(PIP) install torch torchaudio --quiet; \
	else \
		$(PIP) install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet; \
	fi
	@$(PIP) install -e ".[youtube,transcribe,dev]" --quiet
	@$(PIP) install streamlink --quiet
	@echo "✅ kickvoice installed"

install-gpu: venv ## Install kickvoice with CUDA GPU support (Linux only)
	@$(PIP) install torch torchaudio --quiet
	@$(PIP) install -e ".[youtube,transcribe,dev]" --quiet
	@$(PIP) install streamlink --quiet
	@echo "✅ kickvoice installed (GPU)"

download-models: ## Pre-download ML models (~80MB)
	@$(PYTHON) -c "\
		from speechbrain.inference.speaker import EncoderClassifier; \
		import os; \
		EncoderClassifier.from_hparams( \
			source='speechbrain/spkrec-ecapa-voxceleb', \
			savedir=os.path.expanduser('~/.kickvoice/pretrained_models/ecapa'), \
			run_opts={'device': 'cpu'}, \
		); \
		print('✅ ECAPA-TDNN ready')"
	@$(PYTHON) -c "\
		from silero_vad import load_silero_vad; \
		load_silero_vad(); \
		print('✅ Silero VAD ready')"

test: ## Run tests
	@$(PYTHON) -m pytest tests/ -v

verify: ## Verify all imports work
	@$(PYTHON) -c "\
		import torch; print(f'  torch        {torch.__version__}'); \
		import torchaudio; print(f'  torchaudio   {torchaudio.__version__}'); \
		import speechbrain; print(f'  speechbrain  {speechbrain.__version__}'); \
		import streamlink; print(f'  streamlink   {streamlink.__version__}'); \
		import click; print(f'  click        {click.__version__}'); \
		import faster_whisper; print(f'  faster-whisper {faster_whisper.__version__}'); \
		import numpy; print(f'  numpy        {numpy.__version__}'); \
		import kickvoice; print(f'  kickvoice    {kickvoice.__version__}'); \
		print('✅ All OK')"

profiles: ## List enrolled speaker profiles
	@$(KICKVOICE) profiles list

clean: ## Remove pycache and build artifacts
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@rm -rf build/ dist/ *.egg-info 2>/dev/null || true
	@echo "✅ Cleaned"

reset: clean ## Full reset — remove venv, caches, and profiles
	@rm -rf $(VENV)
	@echo "✅ Reset complete (venv removed)"
	@echo "   Speaker profiles at ~/.kickvoice are preserved."
	@echo "   To remove those too: rm -rf ~/.kickvoice"
