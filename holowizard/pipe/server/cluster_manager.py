"""Start the SLURM workers only when work arrives and stop them again when the cluster is idle.

The local Dask scheduler runs for the whole server lifetime (it costs no SLURM resources and keeps
the queue and worker status available). Work that needs workers is wrapped in :meth:`lease`, which
starts the SLURM cluster if needed; a watcher thread stops it after ``idle_seconds`` without leases
and without unfinished tasks in the scheduler.
"""

import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict

from holowizard.core.src.hpc.slurm_cluster import SlurmCluster

# Scheduler task states that mean work is still pending or running.
BUSY_STATES = {"waiting", "processing", "queued", "no-worker"}


class OnDemandCluster:
    def __init__(self, cluster: SlurmCluster, idle_seconds: float, check_seconds: float = 30):
        self.cluster = cluster
        self.idle_seconds = idle_seconds
        self.check_seconds = check_seconds
        self._lock = threading.RLock()
        self._leases = 0
        self._last_active = time.monotonic()
        self._stop = threading.Event()
        cluster.start_scheduler()
        self._watcher = threading.Thread(target=self._watch, name="cluster-idle-watcher", daemon=True)
        self._watcher.start()

    @property
    def running(self) -> bool:
        return self.cluster.cluster is not None

    def ensure_running(self):
        """Start the SLURM workers if they are not running (blocks while the cluster object is created)."""
        with self._lock:
            self._last_active = time.monotonic()
            if not self.running:
                logging.info("Starting SLURM workers on demand.")
                print("Starting SLURM workers on demand.", flush=True)
                self.cluster.start_cluster()

    @contextmanager
    def lease(self):
        """Keep the workers running while the block executes (starting them first if needed)."""
        with self._lock:
            self._leases += 1
        try:
            self.ensure_running()
            yield
        finally:
            with self._lock:
                self._leases -= 1
                self._last_active = time.monotonic()

    def _busy(self) -> bool:
        if self._leases > 0:
            return True
        queue = self.cluster.queue_info()
        return isinstance(queue, list) and any(task.get("state") in BUSY_STATES for task in queue)

    def _watch(self):
        while not self._stop.wait(self.check_seconds):
            try:
                if not self.running:
                    continue
                now = time.monotonic()
                if self._busy():
                    self._last_active = now
                    continue
                if now - self._last_active < self.idle_seconds:
                    continue
                with self._lock:
                    if self._leases == 0 and self.running and time.monotonic() - self._last_active >= self.idle_seconds:
                        print(f"Cluster idle for {self.idle_seconds / 60:g} min, stopping SLURM workers.", flush=True)
                        self.cluster.stop_cluster()
            except Exception as e:
                logging.warning(f"Cluster idle check failed: {e}")

    def status(self) -> Dict[str, Any]:
        try:
            workers = len(self.cluster.client_scheduler.nthreads()) if self.cluster.client_scheduler else 0
        except Exception:
            workers = 0
        idle_for = None if self._busy_cached() else time.monotonic() - self._last_active
        return {
            "running": self.running,
            "workers": workers,
            "idle_minutes": None if idle_for is None else round(idle_for / 60, 1),
            "idle_limit_minutes": round(self.idle_seconds / 60, 1),
        }

    def _busy_cached(self) -> bool:
        # Cheap variant for status pages: leases only, no scheduler round trip.
        return self._leases > 0

    def shutdown(self):
        self._stop.set()
        with self._lock:
            if self.running:
                print("Stopping SLURM workers.", flush=True)
                self.cluster.stop_cluster()
