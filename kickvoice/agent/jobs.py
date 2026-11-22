"""Background jobs for slow agent tools (collection, analysis)."""
from __future__ import annotations

import itertools
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)

Progress = Callable[[str], None]


@dataclass
class Job:
    id: str
    kind: str
    description: str
    status: str = "running"          # running / done / failed
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    finished_at: str | None = None
    progress: list[str] = field(default_factory=list)
    result: Any = None
    error: str | None = None

    def to_dict(self, progress_tail: int = 8) -> dict:
        return {"id": self.id, "kind": self.kind, "description": self.description,
                "status": self.status, "started_at": self.started_at,
                "finished_at": self.finished_at, "progress": self.progress[-progress_tail:],
                "result": self.result, "error": self.error}


class JobRunner:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def submit(self, kind: str, description: str, fn: Callable[[Progress], Any]) -> Job:
        job = Job(id=f"{kind}-{next(self._ids)}", kind=kind, description=description)
        with self._lock:
            self._jobs[job.id] = job

        def progress(msg: str):
            job.progress.append(msg.strip())

        def run():
            try:
                job.result = fn(progress)
                job.status = "done"
            except Exception as e:
                logger.exception("Job %s failed", job.id)
                job.error = str(e) or type(e).__name__
                job.status = "failed"
            finally:
                job.finished_at = datetime.now().isoformat(timespec="seconds")

        threading.Thread(target=run, name=f"kickvoice-job-{job.id}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        return list(self._jobs.values())
