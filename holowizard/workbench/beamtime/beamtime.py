from abc import ABC, abstractmethod
from pathlib import Path
from dask.distributed import as_completed
from holowizard.workbench.scan import Scan
from holowizard.core.src.hpc.slurm_cluster import SlurmCluster as Cluster
from holowizard.workbench.tasks import (
    FlatFieldTask,
    PhaseRetrievalTask,
    FindFocusTask,
    TomographyTask,
)
from holowizard.workbench.utils.submit_and_handle import submit_and_handle
from holowizard.workbench.server.websocket_viewer import WebsocketViewer
from holowizard.workbench.server.websocket_plotter import WebsocketPlotter
import uuid
import random


class Beamtime(ABC):
    """
    Abstract base for beamtime-related objects.

    Attributes:
        path (str): Base path for beamtime or scan data.
        path_raw (str): Path to raw data.
        path_processed (str): Path to processed data.
        meta_dict (dict): Loaded metadata.
    """

    def __init__(self, path_raw, path_processed, log_path, beamtime_name, cluster: Cluster):
        self.beamtime_name = beamtime_name
        self.path_raw = Path(path_raw)
        self.path_processed = Path(path_processed)
        self.log_path = Path(log_path)
        # self.meta_dict = self._load_metadata("")
        self.scans = []
        self.current_scan = None
        self.cluster = cluster

    @abstractmethod
    def _load_metadata(self, path) -> dict:
        """
        Load and return metadata dictionary.
        """
        pass

    def new_scan(self, scan: Scan):
        """
        Add a new scan to the beamtime object.

        Args:
            scan (Scan): Scan object to add.
        """
        if not isinstance(scan, Scan):
            raise TypeError("scan must be an instance of Scan class")
        if scan.a0 is None:
            scan.a0 = scan.get_a0()  # Ensure a0 is set if not provided
        self.scans.append(scan)
        self.run_scan(scan)
        scan.write_html()  # Write HTML after running the scan

    def get_scan(self, scan_name: str) -> Scan:
        """
        Retrieve a scan by its name.

        Args:
            scan_name (str): Name of the scan to retrieve.

        Returns:
            Scan: The scan object if found, otherwise None.
        """
        for scan in self.scans:
            if scan.key == scan_name:
                return scan
        return None

    def run_scan(self, scan: Scan):
        """
        Run the scan pipeline: flatfield → find_focus → reconstruction → tomography.
        """
        # --- FLATFIELD ---
        flatfield_task = FlatFieldTask(scan)
        _, status = submit_and_handle("flatfield", flatfield_task, self.cluster, scan)
        scan.done.append(dict(name="flatfield", status=status))
        if scan.cancelled:
            return

        # --- FIND FOCUS ---
        if "find_focus" in scan.config.scan.tasks:
            find_focus_task = FindFocusTask(scan, flatfield_task.save_path)
            result, status = submit_and_handle("find_focus", find_focus_task, self.cluster, scan)
            if result and status == "done":
                scan.z01 = result.get("z01")
            scan.done.append(dict(name="find_focus", status=status))
            if scan.cancelled:
                return

        # --- RECONSTRUCTION ---
        if "reconstruction" in scan.config.scan.tasks:
            indices = list(range(len(scan.hologram_path)))
            random.shuffle(indices)
            task = PhaseRetrievalTask(scan, flatfield_task.save_path)
            # Finished images, counted here for the worker status page (the scheduler queue only shows the tasks
            # submitted so far and not yet released, so it cannot tell how many images are done).
            scan.reconstruction_done = 0
            scan.reconstruction_failed = 0
            futures = [
                self.cluster.client_scheduler.submit(
                    task,
                    scan,
                    i,
                    key=f"reconstruction-{uuid.uuid4()}-{scan.key}",
                    priority=20,
                )
                for i in indices
            ]
            for f in as_completed(futures):
                try:
                    f.result()
                    scan.reconstruction_done += 1
                except Exception as e:
                    if not scan.cancelled:
                        scan.reconstruction_failed += 1
                    print(f"Reconstruction task failed for scan {scan.key}: {e}")
                finally:
                    f.release()
            status = "cancelled" if scan.cancelled else "done"
            scan.done.append(dict(name="reconstruction", status=status))
            if scan.cancelled:
                return

        # --- TOMOGRAPHY ---
        if "tomography" in scan.config.scan.tasks:
            task = TomographyTask(scan)
            _, status = submit_and_handle("tomography", task, self.cluster, scan)
            scan.done.append(dict(name="tomography", status=status))

    def cancel_task(self, scan_id):
        """
        Cancel a task by its scan ID. First remove all further tasks and then cancel running scan.

        Args:
            scan_id (str): The ID of the scan to cancel.
        """
        for scan in self.scans:
            if scan.key == scan_id:
                scan.cancel()
        queue = self.cluster.queue_info()
        # Cancel tasks in both 'waiting' and 'processing' state
        for task in queue:
            if scan_id in task["key"]:
                future = self.cluster.client_scheduler.futures.get(task["key"])
                if future:
                    future.cancel()
                    print(f"Cancelled task {task['key']} for scan {scan_id}")
                else:
                    print(f"Warning: No future found for task {task['key']}")

    def phase_retrieval_single_holo(self, scan, session_id, find_focus=False, img_name=None, progress=None):
        """
        Optimize a single hologram by running the find focus task.

        Args:
            img_index (int): Index of the image to be optimized.
            progress (callable, optional): called with the name of each step ("flatfield", "find_focus",
                "reconstruction") when it is submitted, e.g. to show the status on the web page.
        """
        report = progress or (lambda step: None)
        report("flatfield")
        flatfield_task = FlatFieldTask(scan)
        _, status = submit_and_handle("flatfield", flatfield_task, self.cluster, scan)
        img_index = [i for i, path in enumerate(scan.hologram_path) if img_name in path][0] if img_name else img_name
        report("find_focus" if find_focus else "reconstruction")
        if find_focus:
            plotter = [WebsocketPlotter(session_id)]
            find_focus_task = FindFocusTask(scan, flatfield_task.save_path, viewer=None, plotter=plotter)
            result, status = submit_and_handle("find_focus", find_focus_task, self.cluster, scan, img_index)
            if result and status == "done":
                scan.z01 = result.get("z01")
                return result
        else:
            viewer = [WebsocketViewer(session_id)]
            task = PhaseRetrievalTask(scan, flatfield_task.save_path, viewer=viewer, save_scratch=True)
            _, status = submit_and_handle("reconstuction", task, self.cluster, scan, img_index)
        return None
