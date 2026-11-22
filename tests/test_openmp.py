"""Intel Macs: PyTorch and CTranslate2 each bring an OpenMP runtime; both must fit in one process."""

import importlib.util
import os
import platform
import subprocess
import sys

import pytest
import torch

from kickvoice import _openmp
from kickvoice._openmp import (
    ENV_VAR,
    allow_duplicate_openmp,
    bundles_duplicate_openmp,
    first_loaded,
    import_torch,
    whisper_cpu_threads,
)

INTEL_MAC = sys.platform == "darwin" and platform.machine() == "x86_64"
HAS_CTRANSLATE2 = importlib.util.find_spec("ctranslate2") is not None
AFFECTED = INTEL_MAC and HAS_CTRANSLATE2  # where kickvoice changes anything at all


def python(code: str, *args: str, timeout: float = 120, **env) -> subprocess.CompletedProcess:
    """Run `code` in a new interpreter whose environment doesn't mention the switch (unless `env` does)."""
    clean = {k: v for k, v in os.environ.items() if k != ENV_VAR}
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**clean, **env},
    )


@pytest.fixture
def torch_threads():
    """Put PyTorch's thread count back afterwards."""
    before = torch.get_num_threads()
    yield before
    torch.set_num_threads(before)


# ── the switch ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "system,machine,installed,expected",
    [
        ("Darwin", "x86_64", True, True),
        ("Darwin", "x86_64", False, False),  # no ctranslate2: only one runtime, nothing to allow
        ("Darwin", "arm64", True, False),  # Apple Silicon wheels don't bundle libiomp5
        ("Linux", "x86_64", True, False),
        ("Windows", "AMD64", True, False),
    ],
)
def test_the_switch_is_set_only_where_two_runtimes_meet(system, machine, installed, expected):
    assert bundles_duplicate_openmp(system, machine) is (system == "Darwin" and machine == "x86_64")
    env: dict[str, str] = {}
    assert allow_duplicate_openmp(env, system, machine, installed) is expected
    assert env == ({ENV_VAR: "TRUE"} if expected else {})


@pytest.mark.parametrize("chosen", ["FALSE", "TRUE", "0", ""])
def test_a_value_the_user_set_is_never_overridden(chosen):
    env = {ENV_VAR: chosen}
    assert allow_duplicate_openmp(env, "Darwin", "x86_64", True) is False
    assert env == {ENV_VAR: chosen}


def test_defaults_to_this_process_and_this_machine(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setattr(_openmp.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(_openmp.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(_openmp, "ctranslate2_installed", lambda: True)
    assert _openmp.two_runtimes() and allow_duplicate_openmp() is True and os.environ[ENV_VAR] == "TRUE"
    monkeypatch.delenv(ENV_VAR)
    monkeypatch.setattr(_openmp.platform, "machine", lambda: "arm64")
    assert not _openmp.two_runtimes() and allow_duplicate_openmp() is False and ENV_VAR not in os.environ


def test_ctranslate2_is_looked_up_without_importing_it(monkeypatch):
    _openmp.ctranslate2_installed.cache_clear()
    assert _openmp.ctranslate2_installed() is HAS_CTRANSLATE2

    def broken(_name):
        raise ValueError("ctranslate2.__spec__ is None")

    _openmp.ctranslate2_installed.cache_clear()
    monkeypatch.setattr(_openmp.importlib.util, "find_spec", broken)
    assert _openmp.ctranslate2_installed() is False  # a half-removed install counts as absent
    _openmp.ctranslate2_installed.cache_clear()


def test_importing_kickvoice_sets_it_before_any_model_library_loads():
    """
    Every kickvoice process starts by importing the package: the CLI, the
    tests, and the MCP server spawned for `monitor` and `extract-profiles`
    (`python -m kickvoice … mcp`), which is where the crash happened.
    """
    proc = python(
        "import os, sys, kickvoice\n"
        f"print(os.environ.get({ENV_VAR!r}))\n"
        "print(sorted({'torch', 'ctranslate2', 'faster_whisper'} & set(sys.modules)))"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["TRUE" if AFFECTED else "None", "[]"]


def test_the_cli_entry_points_import_the_package_first():
    for module in ("kickvoice", "kickvoice.cli"):  # `python -m kickvoice` / run.sh
        proc = python(
            "import os, runpy, sys\n"
            "sys.argv = ['kickvoice', '--help']\n"
            "try:\n"
            f"    runpy.run_module({module!r}, run_name='__main__')\n"
            "except SystemExit:\n"
            "    pass\n"
            f"print('switch:', os.environ.get({ENV_VAR!r}))"
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip().endswith("switch: TRUE" if AFFECTED else "switch: None")


def test_opting_out_is_respected_by_the_real_import():
    proc = python(f"import os, kickvoice; print(os.environ[{ENV_VAR!r}])", **{ENV_VAR: "FALSE"})
    assert proc.stdout.strip() == "FALSE"


# ── load order and threads ───────────────────────────────────────


@pytest.mark.parametrize(
    "modules,expected",
    [
        (["os", "numpy", "ctranslate2", "faster_whisper", "torch"], "ctranslate2"),
        (["os", "torch", "torch.nn", "ctranslate2"], "torch"),
        (["os", "torchaudio", "ctranslate2.specs"], None),  # only the top-level packages count
        ([], None),
    ],
)
def test_first_loaded_reads_the_import_order(modules, expected):
    assert first_loaded(modules) == expected


def test_first_loaded_in_this_process_ignores_blocked_imports(monkeypatch):
    assert first_loaded() in ("torch", "ctranslate2")  # this module imports torch
    others = {name: module for name, module in sys.modules.items() if name != "ctranslate2"}
    monkeypatch.setattr(_openmp.sys, "modules", {"ctranslate2": None, **others})  # None = "import failed"
    assert first_loaded() == "torch"


def test_import_torch_changes_nothing_where_there_is_one_runtime(monkeypatch, torch_threads):
    monkeypatch.setattr(_openmp, "two_runtimes", lambda: False)
    monkeypatch.setattr(_openmp, "first_loaded", lambda: pytest.fail("no need to look"))
    assert import_torch() is torch and torch.get_num_threads() == torch_threads


def test_import_torch_holds_pytorch_to_one_thread_after_ctranslate2(monkeypatch, torch_threads):
    monkeypatch.setattr(_openmp, "two_runtimes", lambda: True)
    monkeypatch.setattr(_openmp, "first_loaded", lambda: "ctranslate2")
    torch.set_num_threads(2)
    assert import_torch() is torch and torch.get_num_threads() == 1
    assert whisper_cpu_threads() is None  # Whisper keeps its threads


def test_pytorch_keeps_its_threads_when_it_was_first_and_whisper_gives_way(monkeypatch, torch_threads):
    monkeypatch.setattr(_openmp, "two_runtimes", lambda: True)
    monkeypatch.setattr(_openmp, "first_loaded", lambda: "torch")
    torch.set_num_threads(2)
    assert import_torch() is torch and torch.get_num_threads() == 2
    assert whisper_cpu_threads() == 1
    monkeypatch.setattr(_openmp, "two_runtimes", lambda: False)
    assert whisper_cpu_threads() is None  # other platforms: faster-whisper's default


ARRANGED = """
import sys
import kickvoice._openmp as openmp

openmp.two_runtimes = lambda: True  # behave as on an Intel Mac
if sys.argv[1] == "blocked":
    sys.modules["ctranslate2"] = None  # importing it now raises ImportError
torch = openmp.import_torch()
print(openmp.first_loaded(), torch.get_num_threads(), openmp.whisper_cpu_threads())
"""


@pytest.mark.skipif(not HAS_CTRANSLATE2, reason="needs ctranslate2 (the transcribe extra)")
def test_import_torch_loads_ctranslate2_first_in_a_fresh_process():
    proc = python(ARRANGED, "normal")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["ctranslate2", "1", "None"]


def test_import_torch_copes_when_ctranslate2_cannot_be_imported():
    proc = python(ARRANGED, "blocked")
    assert proc.returncode == 0, proc.stderr
    loaded, threads, whisper = proc.stdout.split()
    assert (loaded, whisper) == ("torch", "1") and int(threads) >= 1  # torch came first after all


# ── the real libraries, on the machines this is for ──────────────

WORKLOAD = '''
"""A matmul workload in PyTorch and a translation workload in CTranslate2, in a given order."""
import sys

import numpy as np

model_dir, steps = sys.argv[1], sys.argv[2:]


def build_tiny_model():
    """A one-layer random Transformer in CTranslate2's format: no download needed."""
    from ctranslate2.specs import transformer_spec

    d, ffn, vocab = 256, 512, 200
    rng = np.random.default_rng(0)
    spec = transformer_spec.TransformerSpec.from_config(1, 2)
    shapes = {
        "embeddings": (vocab, d), "projection": (vocab, d), "ffn/linear_0": (ffn, d), "ffn/linear_1": (d, ffn),
        "self_attention/linear_0": (3 * d, d), "self_attention/linear_1": (d, d),
        "attention/linear_0": (d, d), "attention/linear_1": (2 * d, d), "attention/linear_2": (d, d),
    }

    def fill(node, path):
        for attr, value in list(vars(node).items()):
            name = f"{path}/{attr}"
            if isinstance(value, list):
                for i, item in enumerate(value):
                    fill(item, f"{name}_{i}") if hasattr(item, "__dict__") else None
            elif hasattr(value, "__dict__") and not isinstance(value, np.ndarray):
                fill(value, name)
            elif value is None and attr in ("gamma", "beta"):
                setattr(node, attr, (np.ones if attr == "gamma" else np.zeros)(d, np.float32))
            elif value is None and attr == "encodings":
                setattr(node, attr, (rng.normal(size=(64, d)) * 0.05).astype(np.float32))
            elif value is None and attr in ("weight", "bias"):
                shape = next(s for key, s in shapes.items() if f"/{key}" in name)
                shape = shape if attr == "weight" else shape[:1]
                setattr(node, attr, (rng.normal(size=shape) * 0.05).astype(np.float32))

    fill(spec.encoder, "encoder")
    fill(spec.decoder, "decoder")
    tokens = ["<blank>", "<s>", "</s>", "<unk>"] + [f"w{i}" for i in range(vocab - 4)]
    spec.register_source_vocabulary(tokens)
    spec.register_target_vocabulary(tokens)
    spec.validate()
    spec.save(model_dir)


for step in steps:
    if step == "plain-torch":  # a host program that imported torch itself, before kickvoice
        import torch

        a = torch.rand(600, 600)
        for _ in range(10):
            a = (a @ a).tanh()
    elif step == "torch":  # the way kickvoice loads it
        import kickvoice
        from kickvoice._openmp import import_torch

        torch = import_torch()
        a = torch.rand(600, 600)
        for _ in range(10):
            a = (a @ a).tanh()
        print("torch", torch.get_num_threads(), flush=True)
    elif step == "whisper":  # stands in for faster-whisper: the same library, threads sized the same way
        import kickvoice
        from kickvoice._openmp import whisper_cpu_threads

        import ctranslate2

        import os
        if not os.path.exists(os.path.join(model_dir, "model.bin")):
            build_tiny_model()
        threads = whisper_cpu_threads() or 4
        translator = ctranslate2.Translator(model_dir, device="cpu", inter_threads=1, intra_threads=threads)
        batch = [[f"w{(i * 7 + j) % 190}" for j in range(30)] for i in range(64)]
        assert len(translator.translate_batch(batch, max_decoding_length=20, beam_size=2)) == 64
        print("whisper", threads, flush=True)
print("done", flush=True)
'''

real_libraries = pytest.mark.skipif(
    not AFFECTED, reason="only Intel Macs with ctranslate2 installed load two OpenMP runtimes"
)


@pytest.mark.slow
@real_libraries
@pytest.mark.parametrize(
    "steps,expected",
    [
        # as `transcribe` and the MCP server do it: the speech model is loaded first
        (["whisper", "torch", "whisper", "torch"], ["whisper 4", "torch 1", "whisper 4", "torch 1", "done"]),
        # a voice is embedded before any speech model is loaded (enrolment, then a live session)
        (["torch", "whisper", "torch", "whisper"], ["torch 1", "whisper 4", "torch 1", "whisper 4", "done"]),
        # torch was already in the process, multi-threaded, before kickvoice was imported
        (["plain-torch", "whisper", "torch", "whisper"], ["whisper 1", None, "whisper 1", "done"]),
    ],
    ids=["speech-model-first", "voice-model-first", "torch-imported-by-the-host"],
)
def test_pytorch_and_ctranslate2_really_run_in_one_process(tmp_path, steps, expected):
    """
    Regression, with the real libraries and nothing exported by hand. Before:
    "OMP: Error #15" and an abort; with only KMP_DUPLICATE_LIB_OK=TRUE, a
    segfault or a hang in the second and third cases.
    """
    model_dir = tmp_path / "tiny-model"
    model_dir.mkdir()
    try:
        proc = python(WORKLOAD, str(model_dir), *steps, timeout=180)
    except subprocess.TimeoutExpired:
        pytest.fail("the two OpenMP runtimes deadlocked")
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr[-2000:]}"
    lines = proc.stdout.split("\n")[:-1]
    assert len(lines) == len(expected)
    for line, want in zip(lines, expected, strict=True):
        assert want is None or line == want  # None: torch keeps whatever thread count the host gave it
