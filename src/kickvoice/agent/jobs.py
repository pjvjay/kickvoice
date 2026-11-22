"""Background jobs for slow agent tools (collection, analysis) — poll with job_status."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

Progress = Callable[[str], None]


@dataclass
class Job:
    id: str
    kind: str
    description: str
    status: str = "queued"  # queued | running | done | error
    progress: deque = field(default_factory=lambda: deque(maxlen=25))
    result: dict | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def to_dict(self) -> dict:
        end = self.finished_at or time.time()
        return {
            "job_id": self.id,
            "kind": self.kind,
            "description": self.description,
            "status": self.status,
            "elapsed_s": round(end - self.started_at, 1),
            "recent_progress": list(self.progress),
            "result": self.result,
            "error": self.error,
        }


class JobRunner:
    def __init__(self, max_workers: int = 2):
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="kv-job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, description: str, fn: Callable[[Progress], dict]) -> Job:
        job = Job(id=uuid.uuid4().hex[:8], kind=kind, description=description)
        with self._lock:
            self._jobs[job.id] = job

        def run():
            job.status = "running"
            try:
                job.result = fn(job.progress.append)
                job.status = "done"
            except Exception as e:  # reported via job_status
                logger.exception("Job %s failed", job.id)
                job.error, job.status = str(e), "error"
            finally:
                job.finished_at = time.time()

        self._pool.submit(run)
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"No job '{job_id}'")
        return job

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def wait(self, job_id: str, timeout: float) -> Job:
        deadline = time.monotonic() + timeout
        job = self.get(job_id)
        while job.status in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(0.05)
        return job

    def shutdown(self):
        self._pool.shutdown(wait=False, cancel_futures=True)
