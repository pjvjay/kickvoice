"""Audio layer: chunking, ffmpeg decoding, VAD."""
import numpy as np
import pytest

from conftest import SR, conversation, silence, tone
from kickvoice.audio.capture import FileCapture, open_capture, rechunk
from kickvoice.audio.ffmpeg import (
    AudioError, decode_to_pcm, pcm_duration_s, pcm_to_numpy, seconds_to_bytes, write_wav,
)
from kickvoice.audio.vad import EnergyVAD


def test_rechunk_non_overlapping():
    data = bytes(range(100))
    assert list(rechunk([data], 10, 10)) == [data[i:i + 10] for i in range(0, 100, 10)]


def test_rechunk_overlapping_windows():
    chunks = list(rechunk([bytes(range(100))], 20, 10))
    assert len(chunks) == 9                      # 0-20, 10-30, ... 80-100
    assert chunks[1] == bytes(range(10, 30))


def test_rechunk_reassembles_small_blocks():
    """Blocks smaller than a chunk are buffered until a full chunk exists."""
    blocks = [bytes([i]) * 3 for i in range(10)]     # 30 bytes total
    joined = b"".join(blocks)
    assert list(rechunk(blocks, 10, 10)) == [joined[0:10], joined[10:20], joined[20:30]]


def test_rechunk_tail_only_when_long_enough():
    assert list(rechunk([bytes(25)], 10, 10, min_tail_bytes=5))[-1] == bytes(5)
    assert len(list(rechunk([bytes(25)], 10, 10, min_tail_bytes=8))) == 2
    assert len(list(rechunk([bytes(25)], 10, 10, min_tail_bytes=0))) == 2


def test_seconds_to_bytes_is_sample_aligned(config):
    assert seconds_to_bytes(1.0, config.audio) == SR * 2
    assert seconds_to_bytes(0.00003, config.audio) % 2 == 0


def test_pcm_numpy_roundtrip():
    samples = pcm_to_numpy(tone(440, 0.1))
    assert len(samples) == int(SR * 0.1)
    assert samples.dtype == np.float32 and -1.0 <= samples.min() and samples.max() <= 1.0


def test_decode_real_file_through_ffmpeg(config, tmp_path):
    """Round-trips through ffmpeg: wav on disk → PCM at the configured rate."""
    src = tmp_path / "clip.wav"
    write_wav(src, tone(440, 2.0), SR)
    pcm = decode_to_pcm(src, config.audio)
    assert pcm_duration_s(pcm, config.audio) == pytest.approx(2.0, abs=0.05)
    # dominant frequency survived the round trip
    spec = np.abs(np.fft.rfft(pcm_to_numpy(pcm)))
    assert np.fft.rfftfreq(len(pcm) // 2, 1 / SR)[spec.argmax()] == pytest.approx(440, abs=5)


def test_decode_missing_and_invalid(config, tmp_path):
    with pytest.raises(AudioError, match="not found"):
        decode_to_pcm(tmp_path / "nope.wav", config.audio)
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"definitely not audio")
    with pytest.raises(AudioError):
        decode_to_pcm(bad, config.audio)


def test_file_capture_chunks_real_audio(config, tmp_path):
    path = tmp_path / "two_voices.wav"
    write_wav(path, conversation([(220, 4), (550, 4)]), SR)
    capture = open_capture(file_path=path, config=config)
    assert isinstance(capture, FileCapture)
    chunks = list(capture.capture_chunks(chunk_duration_s=2, chunk_stride_s=2))
    assert len(chunks) >= 5
    assert all(len(c) == SR * 2 * 2 for c in chunks[:-1])


def test_open_capture_requires_exactly_one_source(config, tmp_path):
    with pytest.raises(ValueError):
        open_capture(config=config)
    with pytest.raises(ValueError):
        open_capture(stream_url="x", file_path=tmp_path, config=config)


def test_energy_vad_finds_speech_and_skips_silence(config):
    pcm = silence(1.0) + tone(300, 3.0) + silence(1.0) + tone(300, 0.4) + silence(1.0)
    segments = EnergyVAD(config.audio).detect_speech(pcm)
    assert len(segments) == 1                      # the 0.4s blip is too short
    seg = segments[0]
    assert seg.duration_s == pytest.approx(3.0, abs=0.2)
    assert seg.start_sample == pytest.approx(SR, abs=SR * 0.1)
    assert len(seg.to_numpy()) == len(seg.pcm_bytes) // 2


def test_energy_vad_on_silence_and_tiny_input(config):
    assert EnergyVAD(config.audio).detect_speech(silence(2.0)) == []
    assert EnergyVAD(config.audio).detect_speech(b"\x00\x00") == []
