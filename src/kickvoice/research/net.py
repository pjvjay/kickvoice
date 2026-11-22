"""
Polite HTTP for research collectors.

Every request to the outside world goes through `Fetcher`, so the rules live
in one place:
  - a descriptive User-Agent (Wikimedia and Reddit require one)
  - a minimum interval between requests to the same host
  - retries with backoff on 429/5xx, honouring Retry-After
  - a disk cache for GETs, so re-running a collection doesn't re-fetch
  - robots.txt is respected for general web pages (not for official APIs)
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import urllib.robotparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

import requests

from kickvoice.config import Config

logger = logging.getLogger(__name__)

MAX_RETRY_WAIT_S = 60.0


class NetError(RuntimeError):
    """A request failed in a way the caller should report."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class RateLimited(NetError):
    """The service asked us to wait longer than we're willing to block."""


class Blocked(NetError):
    """robots.txt disallows fetching this URL."""


@dataclass
class Response:
    url: str
    status: int
    text: str
    headers: dict[str, str]
    from_cache: bool = False

    def json(self) -> Any:
        try:
            return json.loads(self.text)
        except json.JSONDecodeError as e:
            raise NetError(f"Expected JSON from {self.url}, got: {self.text[:200]!r}") from e


class Fetcher:
    def __init__(self, config: Config | None = None, session: requests.Session | None = None):
        self.config = config or Config()
        self.rcfg = self.config.research
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = self.rcfg.user_agent
        self.cache_dir = Path(self.config.cache_dir) / "http"
        self._last_request: dict[str, float] = {}
        self._lock = threading.Lock()
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    # ── public API ───────────────────────────────────────────────

    def get(
        self,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
        cache: bool = True,
        respect_robots: bool = False,
        timeout: float = 30.0,
    ) -> Response:
        full_url = f"{url}?{urlencode(params, doseq=True)}" if params else url
        if respect_robots and not self.allowed_by_robots(full_url):
            raise Blocked(f"robots.txt disallows {full_url}")
        if cache:
            cached = self._cache_read(full_url)
            if cached is not None:
                return cached
        response = self._request("GET", full_url, headers=headers, timeout=timeout)
        if cache and response.status == 200:
            self._cache_write(response)
        return response

    def post(
        self, url: str, data=None, json_body=None, headers=None, auth=None, timeout: float = 60.0
    ) -> Response:
        return self._request(
            "POST", url, data=data, json_body=json_body, headers=headers, auth=auth, timeout=timeout
        )

    def get_json(self, url: str, params: dict | None = None, **kwargs) -> Any:
        response = self.get(url, params=params, **kwargs)
        if response.status != 200:
            raise NetError(f"{url} returned HTTP {response.status}: {response.text[:300]}", response.status)
        return response.json()

    def allowed_by_robots(self, url: str) -> bool:
        parts = urlparse(url)
        root = f"{parts.scheme}://{parts.netloc}"
        if root not in self._robots:
            parser = urllib.robotparser.RobotFileParser()
            try:
                r = self._request("GET", f"{root}/robots.txt", timeout=15)
                if r.status == 200:
                    parser.parse(r.text.splitlines())
                else:  # no robots.txt (404 etc.) → allowed
                    parser.parse([])
            except NetError:
                parser.parse([])
            self._robots[root] = parser
        parser = self._robots[root]
        return parser.can_fetch(self.rcfg.user_agent, url)

    # ── internals ────────────────────────────────────────────────

    def _throttle(self, host: str):
        with self._lock:
            wait = self._last_request.get(host, 0) + self.rcfg.min_request_interval_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_request[host] = time.monotonic()

    def _request(
        self, method, url, data=None, json_body=None, headers=None, auth=None, timeout=30.0, attempts: int = 4
    ) -> Response:
        host = urlparse(url).netloc
        for attempt in range(attempts):
            self._throttle(host)
            try:
                r = self.session.request(
                    method, url, data=data, json=json_body, headers=headers, auth=auth, timeout=timeout
                )
            except requests.RequestException as e:
                if attempt == attempts - 1:
                    raise NetError(f"{method} {url} failed: {e}") from e
                time.sleep(min(2**attempt, MAX_RETRY_WAIT_S))
                continue

            if r.status_code == 429 or r.status_code >= 500:
                wait = _retry_after(r, default=2**attempt)
                if r.status_code == 429 and wait > MAX_RETRY_WAIT_S:
                    raise RateLimited(
                        f"{host} rate limit reached; it asks to wait {wait / 60:.0f} min. "
                        "Re-run later — already-collected items are kept.",
                        429,
                    )
                if attempt < attempts - 1:
                    logger.info("%s → HTTP %d, retrying in %.0fs", host, r.status_code, wait)
                    time.sleep(wait)
                    continue
            return Response(
                url=url,
                status=r.status_code,
                text=r.text,
                headers={k.lower(): v for k, v in r.headers.items()},
            )
        raise NetError(f"{method} {url} failed after {attempts} attempts")

    def _cache_path(self, url: str) -> Path:
        return self.cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()[:32]}.json"

    def _cache_read(self, url: str) -> Response | None:
        path = self._cache_path(url)
        if not path.exists():
            return None
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if age_h > self.rcfg.cache_ttl_hours:
            return None
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return Response(
            url=data["url"], status=200, text=data["text"], headers=data.get("headers", {}), from_cache=True
        )

    def _cache_write(self, response: Response):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        keep = {k: v for k, v in response.headers.items() if k in ("content-type", "last-modified", "date")}
        self._cache_path(response.url).write_text(
            json.dumps({"url": response.url, "text": response.text, "headers": keep})
        )


def _retry_after(r: requests.Response, default: float) -> float:
    """Seconds to wait, from Retry-After or X's x-rate-limit-reset headers."""
    value = r.headers.get("Retry-After")
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    reset = r.headers.get("x-rate-limit-reset")
    if reset:
        try:
            return max(0.0, float(reset) - time.time())
        except ValueError:
            pass
    return float(default)
