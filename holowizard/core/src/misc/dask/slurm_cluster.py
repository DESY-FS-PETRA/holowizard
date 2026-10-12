import os
import sys
import time
import uuid
import threading
import logging
from dask import config
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from holowizard.core.reconstruction.controller.dask_utils.scheduler import start_scheduler
from holowizard.core.parameters.dask_options import DaskOptions


class SlurmCluster():
    def __init__(self, dask_options: DaskOptions, scale_down_workers: bool = True):
        config.set({"distributed.admin.large-graph-warning-threshold": "1000MB"})
        self.dask_options = dask_options
        self.scale_down_workers = scale_down_workers
        self.env = sys.prefix
        self.scheduler = None
        self.client_scheduler = None
        self.worker_thread = None
        self.stop_event = None
        
        logging.info(f"Using environment: {self.env}")

        os.makedirs(self.dask_options.working_dir, exist_ok=True)


    def get_cluster(self) -> SLURMCluster:
        logging.info("Creating Dask SLURM Cluster...")
        cluster = SLURMCluster(
            queue=self.dask_options.partitions,
            cores=1,
            processes=1,
            death_timeout=10,
            memory="128GB",
            walltime=self.dask_options.walltime,
            job_script_prologue=[
                'export LD_PRELOAD=""',
                "source /etc/profile.d/modules.sh",
                "module load maxwell mamba",
                "source activate base",
                ". mamba-init",
                f"mamba activate {self.dask_options.python_env}",
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
                    python -m distributed.cli.dask_worker {self.scheduler_address} --nthreads 1 --memory-limit 128GB --preload holowizard.pipe.cluster.unset &
                done
                """,
            ],
            job_extra_directives=[
                f"--partition={self.dask_options.partitions}",
                f"--output={self.dask_options.working_dir}/slurm-%j.out",
                "--ntasks=1",
                "--nodes=1",
                f'--constraint="{self.dask_options.constraint}"',
                #"--exclude=max-p3ag028"
            ],
            worker_extra_args=["--no-nanny"],
        )

        logging.info(f"Dask Cluster created. Dashboard: {cluster.dashboard_link}")

        return cluster

    def start_scheduler(self):
        # Start scheduler and retrieve address
        if self.scheduler is None:
            self.scheduler, self.scheduler_address = start_scheduler()
        if self.client_scheduler is None and self.scheduler is not None:
            self.client_scheduler: Client = Client(self.scheduler_address)
            logging.info(f"Scheduler address: {self.client_scheduler.dashboard_link}")

    def start_cluster(self) -> None:
        self.start_scheduler()
        self.start_maintainer_thread()
        self.cluster = self.get_cluster()
        self.cluster.scale(jobs=self.dask_options.num_worker)
        logging.info("Waiting for workers to connect...")

    def start_maintainer_thread(self):
        if self.worker_thread is not None:
            return
        self.stop_event = threading.Event()
        # Launch background thread to maintain worker count
        self.worker_thread = threading.Thread(target=self.maintain_workers, daemon=True)
        self.worker_thread.start()

    def maintain_workers(self) -> None:
        while not self.stop_event.is_set():
            time.sleep(self.dask_options.check_interval)
            if self.cluster is None:
                logging.info("No active cluster")
                continue

            scale_up_to = self.dask_options.num_worker
            worker_count = len(self.client_scheduler.scheduler_info()["workers"])
            logging.info(f"{worker_count} workers connected.")
            logging.info(f"Scaling up workers to {scale_up_to}.")
            self.cluster.scale(jobs=scale_up_to)

    def stop_cluster(self) -> None:
        #if self.client_cluster:
        #    self.client_cluster.close()
        if self.cluster:
            self.cluster.scale(0)
            self.cluster.close()
            self.cluster = None
        logging.info("Dask cluster shutdown complete.")

    def cleanup(self) -> None:
        self.stop_cluster()
        self.stop_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join()
        logging.info("Cleanup complete.")

    def queue_info(self):
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
        

    def submit_task_get_future(self, task, args, key=0):
        future = self.client_scheduler.submit(
            task,
            self.dask_options.to_json(),
            args,
            priority=100,
            key=f"{task.__name__}-{uuid.uuid4()}-{key}",
            pure=False,
        )

        return future

    def submit_sirt_task_get_future(self, task, slices_indices, tomogram_path, root, updated_keys_list, angles, n_iters, cor, part, key=0):
        future = self.client_scheduler.submit(
            task,
            slices_indices,
            tomogram_path,
            root,
            updated_keys_list,
            angles,
            n_iters,
            cor,
            part,
            priority=100,
            key=f"{task.__name__}-{uuid.uuid4()}-{key}",
            pure=False,
        )
        return future
    
    def submit_forward_projection_task_get_future(self, task, tomogram_path, output_dir, angle_keys, angles, cor, key=0):
        future = self.client_scheduler.submit(
            task,
            tomogram_path,
            output_dir,
            angle_keys,
            angles,
            cor,
            priority=100,
            key=f"{task.__name__}-{uuid.uuid4()}-{key}",
            pure=False,
        )
        return future