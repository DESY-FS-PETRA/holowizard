import os
import sys
import json
import time
import threading
import logging
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, Scheduler
from holowizard.pipe.cluster.scheduler import start_scheduler
from holowizard.pipe.cluster.cluster import Cluster
from dask.distributed import WorkerPlugin


class SlurmCluster(Cluster):
    """
    Manages a Dask SLURM cluster using parameters from a holopipe JSON cfg.
    Supports dynamic scaling and background worker monitoring.
    """

    def __init__(self, cfg):
        """
        Initialize the manager from a JSON cfg file.

        Args:
            cfg_path (str): Path to holopipe_cfg.json.
        """

        self.dask_cfg = cfg.cluster
        self.env = sys.prefix
        logging.info(f"Using environment: {self.env}")

        self.num_slurm_workers = self.dask_cfg.get("num_slurm_workers", 120)
        self.check_interval = self.dask_cfg.get("check_interval", 120)
        self.slurm_output_path = cfg.paths.get("slurm_dir", "./slurm")
        os.makedirs(self.slurm_output_path, exist_ok=True)

        # Start scheduler and retrieve address
        self.scheduler, self.scheduler_address = start_scheduler()
        self.min_worker = 0
        self.cluster = None
        self.client_cluster = None
        self.client_scheduler = None
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.create_cluster()
        self.start_cluster()
        logging.info(f"Scheduler address: {self.client_scheduler.dashboard_link}")

    def create_cluster(self) -> None:
        """
        Configure and initialize the SLURM cluster.
        Cluster options are read from the config.
        """
        partitions = self.dask_cfg.partitions
        partition_string = ",".join(partitions)

        logging.info("Creating Dask SLURM Cluster...")
        self.cluster = SLURMCluster(
            queue=self.dask_cfg.get("queue", "allgpu"),
            cores=1,
            processes=1,
            death_timeout=5,
            memory=self.dask_cfg.get("memory", "256GB"),
            walltime=self.dask_cfg.get("walltime", "2-00:00:00"),
            job_script_prologue=[
                'export LD_PRELOAD=""',
                "source /etc/profile.d/modules.sh",
                "module load maxwell mamba",
                "source activate base",
                ". mamba-init",
                f"mamba activate {self.env}",
                f"""
                if ! command -v nvidia-smi &> /dev/null; then
                    echo "nvidia-smi not found. Exiting."
                    return 1
                fi
                gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
                echo "Detected $gpu_count GPUs."
                for i in $(seq 1 $gpu_count); do
                gpu_index=$((i - 1))
                CUDA_VISIBLE_DEVICES=$gpu_index \
                    python -m distributed.cli.dask_worker {self.scheduler_address} --nthreads 1 --memory-limit 256GB --preload holowizard.pipe.cluster.unset &
                done
                """,
            ],
            job_extra_directives=[
                f"--partition={partition_string}",
                f"--output={self.slurm_output_path}/slurm-%j.out",
                "--ntasks=1",
                "--nodes=1",
                '--constraint="A100|V100|P100"',
            ],
            worker_extra_args=["--no-nanny"],
        )

        logging.info(f"Dask Cluster created. Dashboard: {self.cluster.dashboard_link}")

    def start_cluster(self) -> None:
        """
        Start the SLURM cluster, launch workers, and connect the Dask client.
        """
        self.create_cluster()
        self.client_cluster = Client(self.cluster)
        self.client_scheduler = Client(self.scheduler_address)
        logging.info(f"Dask client connected to scheduler at {self.client_scheduler.dashboard_link}")

        logging.info("Waiting for workers to connect...")

        worker_count = len(self.client_cluster.scheduler_info()["workers"])
        logging.info(f"{worker_count} workers connected.")
        logging.info(f"Dask dashboard available at: {self.client_cluster.dashboard_link}")

        # Launch background thread to maintain worker count
        self.worker_thread = threading.Thread(target=self.maintain_workers, daemon=True)
        self.worker_thread.start()

    def maintain_workers(self) -> None:
        """
        Keep the number of active workers at the configured target.
        Runs in a background thread.
        """
        while not self.stop_event.is_set():
            jobs = len(self.queue_info())
            if jobs == 0:
                logging.info("No tasks in queue, scaling down workers.")
                self.cluster.scale(self.min_worker)
            else:
                logging.info(f"Scaling up workers to {self.num_slurm_workers}.")
                self.cluster.scale(jobs=min(self.num_slurm_workers, jobs))
            time.sleep(self.check_interval)

    def stop_cluster(self) -> None:
        """
        Stop the cluster, close the client, and clean up threads.
        """
        if self.client_cluster:
            self.client_cluster.close()
        if self.cluster:
            self.cluster.scale(0)
            self.cluster.close()
        self.stop_event.set()
        logging.info("Dask cluster shutdown complete.")

    def cleanup(self) -> None:
        """
        Release Dask resources and stop background threads.
        """
        self.stop_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join()
        logging.info("Cleanup complete.")

    def queue_info(self):
        """
        Retrieve info about tasks currently in the scheduler's task queue.
        Returns:
            dict: Task states (e.g., waiting, processing, memory).
        """
        if not self.client_scheduler:
            return {"error": "Client not connected to scheduler."}

        def inspect(dask_scheduler):
            return [
                {
                    "key": task.key,
                    "state": task.state,
                    "who_has": [str(x) for x in list(task.who_has or "")],
                }
                for task in dask_scheduler.tasks.values()
            ]

        try:
            return self.client_scheduler.run_on_scheduler(inspect)
        except Exception as e:
            logging.error(f"Error retrieving queue info: {e}")
            return {"error": str(e)}
