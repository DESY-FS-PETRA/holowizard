from dataclasses import dataclass
from abc import ABC

from holowizard.core.parameters.dask_options import DaskOptions
from holowizard.core.reconstruction.controller.dask_utils.slurm_cluster import SlurmCluster


@dataclass(init=True)
class AbstractContextDistributed(ABC):
    bunch_size: int
    dask_options: DaskOptions
    cluster: SlurmCluster
