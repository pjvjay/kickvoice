"""
Language model access.

Two backends behind one small interface:
  - "anthropic": the Anthropic Messages API
  - "openai":    any OpenAI-compatible Chat Completions server (OpenAI,
                 or local models via Ollama / vLLM / LM Studio)

`complete_json` is what the rest of the code uses: it asks for JSON, parses
it tolerantly, validates it, and retries once with the validation error so
the model can correct itself.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import numpy as np
import requests

from kickvoice.config import Config

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LLMError(RuntimeError):
    pass


class InvalidOutput(LLMError):
    """The model's reply couldn't be parsed/validated."""


@dataclass
class LLMReply:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


class LLMClient(Protocol):
    def complete(
        self, system: str, prompt: str, model: str, max_tokens: int = 2000, temperature: float = 0.2
    ) -> LLMReply: ...


def _api_key(env_name: str) -> str:
    key = os.environ.get(env_name, "").strip()
    if not key:
        raise LLMError(f"No API key: set the {env_name} environment variable.")
    return key


def _post(url: str, headers: dict, body: dict, timeout: float, attempts: int = 3) -> dict:
    for attempt in range(attempts):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout)
        except requests.RequestException as e:
            if attempt == attempts - 1:
                raise LLMError(f"Model request failed: {e}") from e
            time.sleep(2**attempt)
            continue
        if r.status_code in (429, 500, 502, 503, 529) and attempt < attempts - 1:
            wait = float(r.headers.get("retry-after", 2**attempt) or 1)
            time.sleep(min(wait, 30))
            continue
        if r.status_code != 200:
            raise LLMError(f"Model API returned HTTP {r.status_code}: {r.text[:400]}")
        return r.json()
    raise LLMError("Model request failed after retries")


class AnthropicClient:
    def __init__(self, config: Config):
        cfg = config.llm
        self.base_url = (cfg.base_url or "https://api.anthropic.com").rstrip("/")
        self.key_env = cfg.api_key_env or "ANTHROPIC_API_KEY"
        self.timeout = cfg.timeout_s

    def complete(self, system, prompt, model, max_tokens=2000, temperature=0.2) -> LLMReply:
        started = time.monotonic()
        data = _post(
            f"{self.base_url}/v1/messages",
            headers={
                "x-api-key": _api_key(self.key_env),
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            body={
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=self.timeout,
        )
        text = "".join(
            block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
        )
        usage = data.get("usage", {})
        return LLMReply(
            text=text,
            model=data.get("model", model),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            latency_s=time.monotonic() - started,
        )


class OpenAICompatibleClient:
    def __init__(self, config: Config):
        cfg = config.llm
        self.base_url = (cfg.base_url or "https://api.openai.com/v1").rstrip("/")
        self.key_env = cfg.api_key_env or "OPENAI_API_KEY"
        self.timeout = cfg.timeout_s

    def complete(self, system, prompt, model, max_tokens=2000, temperature=0.2) -> LLMReply:
        started = time.monotonic()
        # Local servers (Ollama etc.) usually don't need a key.
        key = os.environ.get(self.key_env, "").strip()
        headers = {"content-type": "application/json"}
        if key:
            headers["authorization"] = f"Bearer {key}"
        data = _post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            body={
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            },
            timeout=self.timeout,
        )
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected response shape: {str(data)[:300]}") from e
        usage = data.get("usage") or {}
        return LLMReply(
            text=text,
            model=data.get("model", model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_s=time.monotonic() - started,
        )


def make_llm(config: Config) -> LLMClient:
    provider = config.llm.provider.lower()
    if provider == "anthropic":
        return AnthropicClient(config)
    if provider in ("openai", "openai-compatible", "ollama"):
        return OpenAICompatibleClient(config)
    raise LLMError(f"Unknown llm.provider '{config.llm.provider}' (use anthropic or openai)")


# ── JSON helpers ─────────────────────────────────────────────────

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating code fences and chatter."""
    text = text.strip()
    for candidate in (text, *(m.group(1).strip() for m in _FENCE.finditer(text))):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise InvalidOutput(f"Reply wasn't valid JSON: {text[:200]!r}")


def complete_json(
    client: LLMClient,
    system: str,
    prompt: str,
    model: str,
    validate: Callable[[Any], T],
    max_tokens: int = 2000,
    retries: int = 1,
) -> tuple[T, LLMReply]:
    """
    Ask for JSON and validate it. `validate` returns the cleaned value or
    raises ValueError explaining what's wrong; that message is sent back to
    the model for one retry.
    """
    attempt_prompt = prompt
    for attempt in range(retries + 1):
        reply = client.complete(system, attempt_prompt, model, max_tokens=max_tokens)
        try:
            return validate(parse_json(reply.text)), reply
        except (InvalidOutput, ValueError, TypeError, KeyError) as e:
            if attempt == retries:
                raise InvalidOutput(f"Model output failed validation: {e}") from e
            logger.info("Invalid model output (%s); retrying", e)
            attempt_prompt = (
                f"{prompt}\n\nYour previous reply was invalid: {e}\n"
                "Reply again with ONLY valid JSON in the required format."
            )
    raise InvalidOutput("unreachable")


# ── embeddings (optional) ────────────────────────────────────────


class EmbeddingClient:
    """OpenAI-compatible /embeddings (OpenAI, Ollama, vLLM, …)."""

    def __init__(self, config: Config):
        cfg = config.llm
        self.model = cfg.embedding_model
        self.base_url = (cfg.embedding_base_url or cfg.base_url or "https://api.openai.com/v1").rstrip("/")
        self.key_env = cfg.embedding_api_key_env or "OPENAI_API_KEY"
        self.timeout = cfg.timeout_s

    def embed(self, texts: list[str], batch: int = 64) -> np.ndarray:
        out = []
        key = os.environ.get(self.key_env, "").strip()
        headers = {"content-type": "application/json"}
        if key:
            headers["authorization"] = f"Bearer {key}"
        for i in range(0, len(texts), batch):
            data = _post(
                f"{self.base_url}/embeddings",
                headers,
                {"model": self.model, "input": texts[i : i + batch]},
                self.timeout,
            )
            rows = sorted(data["data"], key=lambda d: d["index"])
            out += [row["embedding"] for row in rows]
        vectors = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms


def make_embedder(config: Config) -> EmbeddingClient | None:
    return EmbeddingClient(config) if config.llm.embedding_model else None
