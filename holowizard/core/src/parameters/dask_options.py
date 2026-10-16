import json
import logging
from pathlib import Path


class DaskOptions:
    def __init__(
        self,
        working_dir="",
        num_worker=1,
        partitions="maxgpu,psgpu,allgpu",
        constraint="V100|P100|A100|A6000",
        walltime="07-00:00:00",
        python_env=None,
        check_interval=120,
    ):
        self.working_dir = working_dir
        self.num_worker = num_worker
        self.partitions = partitions
        self.constraint = constraint
        self.walltime = walltime
        self.python_env = python_env
        self.check_interval = check_interval

    def to_json(self):
        class JsonInput:
            def __init__(self, dask_options: DaskOptions):
                self.working_dir = str(dask_options.working_dir)
                self.num_worker = float(dask_options.num_worker)
                self.partitions = dask_options.partitions
                self.constraint = dask_options.constraint
                self.walltime = dask_options.walltime
                self.python_env = dask_options.python_env
                self.check_interval = float(dask_options.check_interval)

        json_writable = JsonInput(self)
        return json.dumps(json_writable, default=lambda o: o.__dict__)

    @property
    def working_dir(self) -> Path:
        return Path(self._working_dir)

    @working_dir.setter
    def working_dir(self, value):
        self._working_dir = Path(value)

    @staticmethod
    def from_dict(data):
        if not isinstance(data, dict):
            logging.error("Data is not a dictionary")
            return None

        if (
            "working_dir" not in data
            or "num_worker" not in data
            or "partitions" not in data
            or "constraint" not in data
            or "walltime" not in data
            or "python_env" not in data
            or "check_interval" not in data
        ):
            logging.error("Missing required keys in data")
            return None

        return DaskOptions(
            working_dir=data["working_dir"],
            num_worker=int(data["num_worker"]),
            partitions=data["partitions"],
            constraint=data["constraint"],
            walltime=data["walltime"],
            python_env=data["python_env"],
            check_interval=int(data["check_interval"]),
        )

    @staticmethod
    def from_json(data):
        obj_dict = json.loads(data)
        return DaskOptions.from_dict(obj_dict)
