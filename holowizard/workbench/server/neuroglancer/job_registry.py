"""The conversions of this server process."""

import threading
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from holowizard.workbench.server.neuroglancer.zarr_job import ZarrJob

RUNNING = ("reading", "converting")


class JobRegistry:
    """Conversion jobs by id, and the latest job per cache entry and per user (thread safe)."""

    def __init__(self):
        self._jobs: Dict[str, "ZarrJob"] = {}  # job id -> job
        self._cache_jobs: Dict[str, str] = {}  # cache id -> latest job id
        self._user_jobs: Dict[str, str] = {}  # user -> latest job id
        self._lock = threading.Lock()

    def add(self, job: "ZarrJob"):
        with self._lock:
            self._jobs[job.id] = job
            self._cache_jobs[job.cache_id] = job.id
            self._user_jobs[job.user] = job.id

    def get(self, job_id: str) -> Optional["ZarrJob"]:
        with self._lock:
            return self._jobs.get(job_id)

    def latest_for_cache(self, cache_id: str) -> Optional["ZarrJob"]:
        with self._lock:
            return self._jobs.get(self._cache_jobs.get(cache_id, ""))

    def latest_per_cache(self) -> Dict[str, "ZarrJob"]:
        with self._lock:
            return {cache_id: self._jobs[job_id] for cache_id, job_id in self._cache_jobs.items()}

    def running_conflict(self, cache_id: str, user: str) -> Optional["ZarrJob"]:
        """A running conversion of this cache entry or of this user (one at a time each), or None."""
        with self._lock:
            for job_id in (self._cache_jobs.get(cache_id), self._user_jobs.get(user)):
                job = self._jobs.get(job_id or "")
                if job is not None and job.status in RUNNING:
                    return job
        return None


registry = JobRegistry()
