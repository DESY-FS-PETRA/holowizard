import logging
from tqdm import tqdm
from typing import List
from dask.distributed import as_completed

import holowizard.core
from holowizard.core.reconstruction.controller.dask_utils.slurm_cluster import SlurmCluster
from holowizard.core.reconstruction.contexts.context_projection_multistage import ContextProjectionMultistage
from holowizard.core.reconstruction.contexts.context_projection_singlestage import ContextProjectionSingleStage

def broadcast_task(
    cluster: SlurmCluster,
    projection_contexts: List[ContextProjectionMultistage],
    task,
    keys=None,
    device=holowizard.core.torch_running_device,
    max_retries: int = 3,
) -> List[ContextProjectionSingleStage]:
    results = []

    if keys is None:
        keys = [projection_context.current_projection_singlestage.key for projection_context in projection_contexts]

    logging.info("Broadcasting task")

    # map key -> the projection_context it belongs to, so we can resubmit later
    context_by_key = {}
    future_to_key = {}
    retries_by_key = {}

    pbar = tqdm(total=len(keys))
    for key in keys:
        for projection_context in projection_contexts:
            if projection_context.key == key:
                context_by_key[key] = projection_context
                future = cluster.submit_task_get_future(task, projection_context.current_projection_singlestage.to_json())
                future_to_key[future] = key
                retries_by_key[key] = 0
                results.append(projection_context.current_projection_singlestage)
                pbar.update(1)
                break
    pbar.close()

    logging.info("Reading results")

    pbar = tqdm(total=len(keys))
    ac = as_completed(list(future_to_key.keys()))

    for future in ac:
        key = future_to_key[future]
        try:
            result = future.result()
        except Exception as e:
            retries_by_key[key] += 1
            if retries_by_key[key] <= max_retries:
                logging.warning(
                    f"Task for key {key} failed with {e!r}, retrying "
                    f"({retries_by_key[key]}/{max_retries})"
                )
                projection_context = context_by_key[key]
                new_future = cluster.submit_task_get_future(
                    task, projection_context.current_projection_singlestage.to_json()
                )
                future_to_key[new_future] = key
                ac.add(new_future)
            else:
                logging.error(f"Task for key {key} failed after {max_retries} retries: {e!r}")
                pbar.update(1)
            continue

        if result is not None:
            json_data = str(result)
            context = ContextProjectionSingleStage.from_json(json_data, device)
            results.append(context)
        pbar.update(1)

    pbar.close()
    return results
