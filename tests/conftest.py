"""Shared fixtures: isolated config dirs + model-free fakes for the pipeline."""
import math
import struct

import numpy as np
import pytest

from kickvoice.config import Config
from kickvoice.speakers.encoder import SpeakerEncoder

SR = 16000
DIM = 192


@pytest.fixture
def config(tmp_path):
    cfg = Config(data_dir=tmp_path / "data", profiles_dir=tmp_path / "data" / "profiles")
    cfg.extract.min_speaker_speech_s = 8.0
    cfg.extract.chunk_duration_s = 10.0
    return cfg


def voice_vector(freq: float) -> np.ndarray:
    """Stable pseudo-'voiceprint' for a synthetic voice (tone frequency)."""
    rng = np.random.default_rng(int(freq))
    v = rng.normal(size=DIM)
    return v / np.linalg.norm(v)


def tone(freq: float, seconds: float, amp: float = 0.3) -> bytes:
    n = int(SR * seconds)
    return struct.pack(
        f"<{n}h", *(int(amp * 32767 * math.sin(2 * math.pi * freq * i / SR)) for i in range(n))
    )


def silence(seconds: float) -> bytes:
    return b"\x00\x00" * int(SR * seconds)


def conversation(turns: list[tuple[float, float]], gap: float = 0.6) -> bytes:
    """[(freq, seconds), ...] → PCM with silence between turns."""
    out = [silence(gap)]
    for freq, secs in turns:
        out += [tone(freq, secs), silence(gap)]
    return b"".join(out)


class FakeEncoder(SpeakerEncoder):
    """
    Real encoder API, no model: maps a window to its dominant frequency's
    voiceprint (+ a little noise). Subclassing keeps it honest — if the
    encoder API changes, the fake changes with it.
    """

    def __init__(self, noise: float = 0.15, seed: int = 0, voices: dict | None = None):
        super().__init__()
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.calls = 0
        self.voices = voices or {}          # freq → custom voiceprint (sound-alikes)

    def embed_pcm(self, pcm: bytes) -> np.ndarray:
        self.calls += 1
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
        spec = np.abs(np.fft.rfft(x))
        freq = np.fft.rfftfreq(len(x), 1 / SR)[spec.argmax()]
        bucket = round(freq / 10) * 10          # tolerate FFT bin jitter
        base = self.voices.get(bucket)
        v = (voice_vector(bucket) if base is None else base) + self.rng.normal(scale=self.noise / math.sqrt(DIM), size=DIM)
        return v / np.linalg.norm(v)


class FakeCapture:
    """Serves PCM in blocks like StreamCapture/FileCapture, tracking closure."""

    def __init__(self, pcm: bytes):
        self.pcm = pcm
        self.closed = False
        self.requested = None

    def capture_chunks(self, chunk_duration_s=None, chunk_stride_s=None):
        self.requested = (chunk_duration_s, chunk_stride_s)
        size = int(SR * 2 * chunk_duration_s)
        try:
            for off in range(0, len(self.pcm), size):
                yield self.pcm[off:off + size]
        finally:
            self.closed = True


class ScriptedInput:
    """Feeds canned answers to Reviewer prompts; records what was asked."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


class FakeRecognizer:
    """
    ASR stand-in with Whisper's output shape: ~0.5s "words" over loud audio,
    each named after the dominant pitch ("w220"), so tests can check which
    voice every word was attributed to. `hallucinate=True` also emits a word
    in the middle of every silence, like Whisper does over music/silence.
    """

    model_name = "fake-asr"

    def __init__(self, word_s: float = 0.5, hallucinate: bool = False):
        self.word_s = word_s
        self.hallucinate = hallucinate
        self.calls: list[float] = []          # duration of each audio passed in

    def transcribe(self, audio, sample_rate):
        from kickvoice.transcripts.asr import Word

        self.calls.append(len(audio) / sample_rate)
        frame = int(sample_rate * 0.03)
        n = len(audio) // frame
        loud = np.sqrt((audio[:n * frame].reshape(n, frame) ** 2).mean(axis=1)) > 0.02
        padded = np.concatenate(([False], loud, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        words, prev_end = [], 0.0
        for a, b in zip(edges[::2], edges[1::2]):
            start, end = a * 0.03, b * 0.03
            if self.hallucinate and start - prev_end > 0.5:
                mid = (prev_end + start) / 2
                words.append(Word(mid - 0.1, mid + 0.1, " thanks", 0.3))
            t = start
            while t < end - 0.05:
                w_end = min(t + self.word_s, end)
                seg = audio[int(t * sample_rate):int(w_end * sample_rate)]
                spec = np.abs(np.fft.rfft(seg))
                freq = round(np.fft.rfftfreq(len(seg), 1 / sample_rate)[spec.argmax()] / 10) * 10
                words.append(Word(t, w_end, f" w{int(freq)}", 0.9))
                t = w_end
            prev_end = end
        return words


# ── research fakes ───────────────────────────────────────────────

import json as _json
import threading as _threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


class FakeWeb:
    """
    A local HTTP server standing in for Wikipedia, X, Reddit, LLM APIs, etc.
    Routes: {(METHOD, path): handler(req) -> (status, headers, body)} where
    body may be a dict (sent as JSON) or str. Every request is recorded.
    """

    def __init__(self):
        self.routes = {}
        self.requests = []
        web = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _handle(self, method):
                parsed = urlparse(self.path)
                length = int(self.headers.get("content-length") or 0)
                body = self.rfile.read(length).decode() if length else ""
                req = {"method": method, "path": parsed.path,
                       "query": {k: v[0] for k, v in parse_qs(parsed.query).items()},
                       "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body}
                web.requests.append(req)
                handler = web.routes.get((method, parsed.path))
                if handler is None:
                    status, headers, payload = 404, {}, {"error": "not found"}
                else:
                    status, headers, payload = handler(req)
                data = _json.dumps(payload) if isinstance(payload, (dict, list)) else str(payload)
                ctype = "application/json" if isinstance(payload, (dict, list)) else "text/html"
                self.send_response(status)
                self.send_header("content-type", headers.pop("content-type", ctype))
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("content-length", str(len(data.encode())))
                self.end_headers()
                self.wfile.write(data.encode())

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self._thread = _threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()

    def route(self, method, path, handler=None, *, json=None, status=200, headers=None, text=None):
        if handler is None:
            payload = json if json is not None else (text or "")
            handler = lambda req, p=payload: (status, dict(headers or {}), p)  # noqa: E731
        self.routes[(method, path)] = handler

    def hits(self, path):
        return [r for r in self.requests if r["path"] == path]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake_web():
    web = FakeWeb()
    yield web
    web.close()


@pytest.fixture
def rconfig(config, fake_web):
    """Config whose research endpoints all point at the fake web server."""
    r = config.research
    r.min_request_interval_s = 0.0
    r.wikipedia_api = f"{fake_web.url}/w/api.php"
    r.wikidata_api = f"{fake_web.url}/wd/api.php"
    r.x_api = f"{fake_web.url}/2"
    r.reddit_api = f"{fake_web.url}/reddit"
    r.reddit_auth = f"{fake_web.url}/reddit/api/v1/access_token"
    r.brave_search_api = f"{fake_web.url}/brave"
    return config


class ScriptedLLM:
    """LLM stand-in: replies come from a list, or from a function of the prompt."""

    def __init__(self, replies=None, fn=None, model="fake-model"):
        self.replies = list(replies or [])
        self.fn = fn
        self.model = model
        self.calls = []

    def complete(self, system, prompt, model, max_tokens=2000, temperature=0.2):
        from kickvoice.research.llm import LLMReply

        self.calls.append({"system": system, "prompt": prompt, "model": model})
        if self.fn is not None:
            text = self.fn(system, prompt)
        else:
            text = self.replies.pop(0)
        if not isinstance(text, str):
            text = _json.dumps(text)
        return LLMReply(text=text, model=self.model, input_tokens=len(prompt) // 4,
                        output_tokens=len(text) // 4)
