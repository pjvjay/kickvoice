"""
Intel Macs: make PyTorch and CTranslate2 work in one process.

On macOS x86_64 the PyTorch wheel (speaker embeddings, Silero VAD) and the
CTranslate2 wheel (faster-whisper, speech-to-text) each bundle their own copy
of Intel's OpenMP runtime, libiomp5. Measured on an Intel Mac with torch 2.2.2
and ctranslate2 4.9.0, running a real workload in each library:

    as installed, either load order         abort: "OMP: Error #15: Initializing
                                            libiomp5.dylib, but found
                                            libiomp5.dylib already initialized."
    KMP_DUPLICATE_LIB_OK=TRUE, and the      segfault or deadlock
    library loaded second uses threads
    KMP_DUPLICATE_LIB_OK=TRUE, and the      works, in both orders
    library loaded second uses one thread

The abort killed every command that loads both models: `transcribe`, `debate`,
and the MCP server behind `monitor` and `extract-profiles`, where the terminal
only showed "MCPError: Connection closed". Exporting the variable by hand
appeared to fix it only because of two accidents: the Whisper model happens
to be loaded first, and importing silero-vad happens to set PyTorch to one
thread.

kickvoice now arranges the working case itself, on macOS x86_64 with
ctranslate2 installed, and does nothing anywhere else:

1. `import kickvoice` sets KMP_DUPLICATE_LIB_OK=TRUE unless it is already
   set. The runtime reads it when it initialises, so this has to happen
   before either library is imported; every kickvoice process (the CLI, the
   MCP server it spawns) starts by importing the package.
2. `import_torch()`, which kickvoice uses wherever it first touches torch,
   loads CTranslate2 first and then holds PyTorch to one thread. Whisper is
   by far the heavier model, so it is the one that keeps its threads.
3. If PyTorch was already loaded first (a host program imported it before
   kickvoice), the roles swap: `whisper_cpu_threads()` tells the recogniser
   to run Whisper on one thread.

Intel documents KMP_DUPLICATE_LIB_OK as an unsupported workaround and does
not rule out crashes or wrong results. Export KMP_DUPLICATE_LIB_OK=FALSE to
get the abort back instead; a value you set is never overridden.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import platform
import sys
from collections.abc import Iterable, MutableMapping
from functools import lru_cache

ENV_VAR = "KMP_DUPLICATE_LIB_OK"
_OPENMP_LIBRARIES = ("torch", "ctranslate2")


def bundles_duplicate_openmp(system: str | None = None, machine: str | None = None) -> bool:
    """True on macOS x86_64, where both wheels ship their own libiomp5."""
    system = platform.system() if system is None else system
    machine = platform.machine() if machine is None else machine
    return system == "Darwin" and machine == "x86_64"


@lru_cache(maxsize=1)
def ctranslate2_installed() -> bool:
    """Whether ctranslate2 can be imported (checked without importing it)."""
    try:
        return importlib.util.find_spec("ctranslate2") is not None
    except (ImportError, ValueError):  # a broken or half-removed install
        return False


def two_runtimes() -> bool:
    """True where PyTorch and CTranslate2 would each bring an OpenMP runtime into this process."""
    return bundles_duplicate_openmp() and ctranslate2_installed()


def allow_duplicate_openmp(
    environ: MutableMapping[str, str] | None = None,
    system: str | None = None,
    machine: str | None = None,
    installed: bool | None = None,
) -> bool:
    """
    Set KMP_DUPLICATE_LIB_OK=TRUE where it's needed, unless the user set it.

    Returns True if this call set it.
    """
    environ = os.environ if environ is None else environ
    installed = ctranslate2_installed() if installed is None else installed
    if ENV_VAR in environ or not installed or not bundles_duplicate_openmp(system, machine):
        return False
    environ[ENV_VAR] = "TRUE"
    return True


def first_loaded(modules: Iterable[str] | None = None) -> str | None:
    """Which of torch and ctranslate2 was imported first in this process (None if neither was)."""
    if modules is None:  # sys.modules keeps import order; None marks a blocked or failed import
        modules = [name for name, module in list(sys.modules.items()) if module is not None]
    return next((name for name in modules if name in _OPENMP_LIBRARIES), None)


def import_torch():
    """
    `import torch`, arranged so that it can share the process with CTranslate2.

    Use this instead of a bare import wherever kickvoice first touches torch.
    Where two OpenMP runtimes would meet, it loads CTranslate2 first and
    holds PyTorch to one thread (see the module docstring for why).
    """
    if not two_runtimes():
        import torch

        return torch
    if first_loaded() is None:
        with contextlib.suppress(ImportError):
            import ctranslate2  # noqa: F401
    import torch

    if first_loaded() == "ctranslate2" and torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    return torch


def whisper_cpu_threads() -> int | None:
    """
    Threads faster-whisper may use: None for its default, or 1 when PyTorch
    got into the process first and CTranslate2 therefore must not start a
    thread team.
    """
    return 1 if two_runtimes() and first_loaded() == "torch" else None
