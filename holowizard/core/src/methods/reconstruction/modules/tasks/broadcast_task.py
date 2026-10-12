from typing import List

import holowizard.core.src
from holowizard.core.src.hpc.slurm_cluster import SlurmCluster
from holowizard.core.src.hpc.broadcast_task import broadcast_task as broadcast_task_impl
from holowizard.core.src.methods.reconstruction.modules.contexts.context_projection_multistage import ContextProjectionMultistage
from holowizard.core.src.methods.reconstruction.modules.contexts.context_projection_singlestage import ContextProjectionSingleStage

def broadcast_task(
    cluster: SlurmCluster,
    projection_contexts: List[ContextProjectionMultistage],
    task,
    keys=None,
    device=holowizard.core.src.torch_running_device,
) -> List[ContextProjectionSingleStage]:
    results = broadcast_task_impl(
    cluster,
    [c.current_projection_singlestage for c in projection_contexts],
    task,
    keys=keys,
    from_json_kwargs={"device": device},
    )

    return results