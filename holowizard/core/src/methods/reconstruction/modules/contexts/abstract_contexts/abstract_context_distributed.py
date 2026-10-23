from dataclasses import dataclass
from abc import ABC

from holowizard.core.src.parameters.dask_options import DaskOptions
from holowizard.core.src.methods.reconstruction.controller.dask_utils.slurm_cluster import SlurmCluster


@dataclass(init=True)
class AbstractContextDistributed(ABC):
    bunch_size: int
    dask_options: DaskOptions
    cluster: SlurmCluster
