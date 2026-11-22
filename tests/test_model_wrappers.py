"""
Wrappers around third-party models (Silero VAD, SpeechBrain ECAPA, faster-whisper).

The real models need downloads, so these tests install stand-in modules with
the same API and check *our* logic: argument passing, segment filtering,
shapes, fallbacks and error messages.
"""

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from conftest import SR, conversation, tone
from kickvoice.audio.ffmpeg import write_wav
from kickvoice.audio.vad import EnergyVAD, SileroVAD, get_vad
from kickvoice.speakers.encoder import SpeakerEncoder
from kickvoice.transcripts.asr import ASRUnavailable, FasterWhisperRecognizer

# ── Silero VAD ───────────────────────────────────────────────────


def fake_silero(monkeypatch, timestamps, seen):
    module = types.ModuleType("silero_vad")
    module.load_silero_vad = lambda: "model"

    def get_speech_timestamps(audio, model, **kw):
        seen.update(kw, audio_len=len(audio), model=model)
        return timestamps

    module.get_speech_timestamps = get_speech_timestamps
    monkeypatch.setitem(sys.modules, "silero_vad", module)


def test_silero_vad_filters_short_segments(config, monkeypatch):
    seen = {}
    fake_silero(monkeypatch, [{"start": 0, "end": SR * 2}, {"start": SR * 3, "end": SR * 3 + 800}], seen)
    vad = SileroVAD(config.audio)
    pcm = tone(220, 4)
    segments = vad.detect_speech(pcm, threshold=0.4, min_duration_s=1.0)
    assert [(s.start_sample, s.end_sample) for s in segments] == [(0, SR * 2)]
    assert segments[0].pcm_bytes == pcm[: SR * 4] and segments[0].sample_rate == SR
    assert seen["threshold"] == 0.4 and seen["min_speech_duration_ms"] == 1000
    assert seen["sampling_rate"] == SR and seen["audio_len"] == SR * 4 and seen["model"] == "model"
    vad.detect_speech(pcm)  # defaults from config
    assert seen["threshold"] == config.audio.vad_threshold
    segments[0].to_float_tensor()


def test_silero_vad_torch_hub_fallback(config, monkeypatch):
    monkeypatch.setitem(sys.modules, "silero_vad", None)  # → ImportError
    calls = {}

    def hub_load(**kw):
        calls.update(kw)
        return "hub-model", (lambda audio, model, **k: [{"start": 0, "end": SR * 2}],)

    monkeypatch.setattr(torch.hub, "load", hub_load)
    segments = SileroVAD(config.audio).detect_speech(tone(220, 3))
    assert calls["repo_or_dir"] == "snakers4/silero-vad" and len(segments) == 1


def test_get_vad_selection(config, monkeypatch):
    assert isinstance(get_vad(audio_cfg=config.audio), SileroVAD)
    assert isinstance(get_vad(use_silero=False, audio_cfg=config.audio), EnergyVAD)
    import kickvoice.audio.vad as vad_mod

    monkeypatch.setattr(vad_mod.importlib.util, "find_spec", lambda name: None)
    assert isinstance(get_vad(audio_cfg=config.audio), EnergyVAD)


# ── ECAPA speaker encoder ────────────────────────────────────────


@pytest.fixture
def fake_speechbrain(monkeypatch):
    calls = {}

    class EncoderClassifier:
        @classmethod
        def from_hparams(cls, source, savedir, run_opts):
            calls.update(source=source, savedir=savedir, run_opts=run_opts)
            return cls()

        def encode_batch(self, audio):
            calls.setdefault("shapes", []).append(tuple(audio.shape))
            return torch.ones(1, 1, 192) * audio.abs().mean()

    speaker = types.ModuleType("speechbrain.inference.speaker")
    speaker.EncoderClassifier = EncoderClassifier
    for name, mod in (
        ("speechbrain", types.ModuleType("speechbrain")),
        ("speechbrain.inference", types.ModuleType("speechbrain.inference")),
        ("speechbrain.inference.speaker", speaker),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def test_encoder_loads_once_and_embeds(config, fake_speechbrain, tmp_path):
    enc = SpeakerEncoder(config)
    emb = enc.embed_pcm(tone(220, 2))
    assert emb.shape == (192,)
    assert fake_speechbrain["source"] == config.embedding.model_source
    assert fake_speechbrain["run_opts"] == {"device": "cpu"}
    assert fake_speechbrain["savedir"].endswith("pretrained_models/ecapa")
    assert fake_speechbrain["shapes"][0] == (1, SR * 2)

    seg = EnergyVAD(config.audio).detect_speech(conversation([(220, 2)]))[0]
    assert enc.embed_segment(seg).shape == (192,)
    assert len(enc.embed_segments([seg, seg])) == 2
    path = tmp_path / "clip.wav"
    write_wav(path, tone(220, 1.5), SR)
    assert enc.embed_file(path).shape == (192,)
    assert len(fake_speechbrain["shapes"]) == 5  # model loaded once, used 5×


def test_mean_embedding():
    m = SpeakerEncoder.mean_embedding([np.array([3.0, 0.0]), np.array([0.0, 4.0])])
    assert np.isclose(np.linalg.norm(m), 1.0)
    assert SpeakerEncoder.compute_mean_embedding is SpeakerEncoder.mean_embedding
    assert np.allclose(SpeakerEncoder.mean_embedding([np.zeros(2)]), 0)
    with pytest.raises(ValueError):
        SpeakerEncoder.mean_embedding([])


# ── faster-whisper ───────────────────────────────────────────────


def fake_whisper(monkeypatch, fail_load=False):
    calls = {}

    class WhisperModel:
        def __init__(self, size, device, compute_type, download_root):
            if fail_load:
                raise RuntimeError("CUDA driver not found\nmore details")
            calls.update(size=size, device=device, compute_type=compute_type, root=download_root)

        def transcribe(self, audio, **kw):
            calls.update(kw, dtype=str(audio.dtype))
            words = [
                SimpleNamespace(start=0.0, end=0.4, word=" hello", probability=0.9),
                SimpleNamespace(start=0.4, end=0.5, word="  ", probability=0.1),
                SimpleNamespace(start=0.5, end=1.0, word=" world", probability=0.8),
            ]
            return iter([SimpleNamespace(words=words), SimpleNamespace(words=None)]), {}

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return calls


def test_whisper_recognizer(config, monkeypatch):
    calls = fake_whisper(monkeypatch)
    config.transcribe.language = "en"
    rec = FasterWhisperRecognizer(config)
    assert rec.model_name == "faster-whisper/small"
    assert rec.transcribe(np.zeros(0, dtype=np.float32), SR) == []  # no audio → no model load
    words = rec.transcribe(np.zeros(SR, dtype=np.float64), SR)
    assert [(w.text, w.start, w.probability) for w in words] == [(" hello", 0.0, 0.9), (" world", 0.5, 0.8)]
    assert words[0].mid == pytest.approx(0.2)
    assert calls["word_timestamps"] and calls["language"] == "en" and calls["vad_filter"] is False
    assert calls["condition_on_previous_text"] is False and calls["dtype"] == "float32"
    assert calls["compute_type"] == "int8" and calls["root"].endswith("pretrained_models/whisper")
    rec.load()  # idempotent
    with pytest.raises(ValueError, match="16kHz"):
        rec.transcribe(np.zeros(10, dtype=np.float32), 8000)


def test_whisper_load_failure_is_explained(config, monkeypatch):
    fake_whisper(monkeypatch, fail_load=True)
    with pytest.raises(ASRUnavailable, match="Couldn't load Whisper model 'small': CUDA driver not found"):
        FasterWhisperRecognizer(config).load()


def test_whisper_not_installed(config, monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises(ASRUnavailable, match=r"pip install"):
        FasterWhisperRecognizer(config).load()
