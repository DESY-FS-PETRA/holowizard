class DaskOptions:
    def __init__(
        self,
        working_dir="",
        num_worker=1,
        partitions="maxgpu,psgpu,allgpu",
        constraint="V100|P100|A100|A6000",
        walltime="07-00:00:00",
        python_env=None,
    ):
        self.working_dir = working_dir + "/"
        self.num_worker = num_worker
        self.partitions = partitions
        self.constraint = constraint
        self.walltime = walltime
        self.python_env = python_env
