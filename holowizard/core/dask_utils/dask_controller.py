from pathlib import Path
import shutil
import os
import logging
import threading
import time
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

from holowizard.core.parameters.dask_options import DaskOptions


class DaskController:
    def __init__(self, dask_options: DaskOptions):
        self.lock = threading.Lock()
        self.running = False
        self.stopped = True
        self.stopped_condition = threading.Condition()
        self.dask_options = dask_options

        self.cluster = SLURMCluster(
            log_directory=dask_options.working_dir,
            queue=dask_options.partitions,
            account=None,
            cores=1,
            processes=1,
            memory="256GB",
            walltime=dask_options.walltime,
            job_script_prologue=[
                'export LD_PRELOAD=""',
                "source /etc/profile.d/modules.sh",
                "module load maxwell",
                "module load anaconda/3",
                "source activate " + dask_options.python_env,
            ],
            job_extra_directives=[
                "--time=" + dask_options.walltime,
                "--ntasks=1",
                "--nodes=1",
                "--constraint=" + dask_options.constraint,
                "--exclude=max-exflg007",
            ],
            worker_extra_args=["--no-nanny"],
        )

        self.worker_maintenance_thread = None

    def start(self):
        def maintain_workers(dask_controller: DaskController):
            logging.info("Dask maintainer thread started")
            while True:
                time.sleep(60)
                try:
                    # Get the number of active workers
                    active_workers = len(
                        dask_controller.client.scheduler_info()["workers"]
                    )
                    logging.info(f"Current active workers: {active_workers}")
                    if active_workers < dask_controller.dask_options.num_worker:
                        logging.info(
                            f"Scaling up to {dask_controller.dask_options.num_worker} workers. Current: {active_workers}"
                        )
                        dask_controller.cluster.scale(
                            jobs=dask_controller.dask_options.num_worker
                        )
                except Exception as e:
                    logging.error(f"Error while maintaining workers: {e}")
                if dask_controller.running is False:
                    break

            dask_controller.client.shutdown()
            dask_controller.client.close()
            dask_controller.client = None

            with self.stopped_condition:
                dask_controller.stopped = True
                dask_controller.stopped_condition.notify_all()

        with self.stopped_condition:
            if self.running:
                return

            logging.info("Checking state of dask maintainer thread")
            with self.stopped_condition:
                while not self.stopped:
                    self.stopped_condition.wait()

            self.rmdir(self.dask_options.working_dir)
            if not os.path.isdir(self.dask_options.working_dir):
                os.mkdir(self.dask_options.working_dir)

            logging.info(
                "Starting pool of "
                + str(self.dask_options.num_worker)
                + " dask worker."
            )
            self.cluster.scale(jobs=self.dask_options.num_worker)
            self.client = Client(self.cluster)

            logging.info("Starting dask maintainer thread")
            self.running = True
            self.stopped = False

            self.worker_maintenance_thread = threading.Thread(
                target=maintain_workers, daemon=True, args=(self,)
            )
            self.worker_maintenance_thread.start()

    def stop(self):
        with self.stopped_condition:
            self.running = False
            while not self.stopped:
                self.stopped_condition.wait()

    def rmdir(self, directory):
        shutil.rmtree(directory, ignore_errors=True)
        directory = Path(directory)
        while directory.exists():
            time.sleep(0.1)
