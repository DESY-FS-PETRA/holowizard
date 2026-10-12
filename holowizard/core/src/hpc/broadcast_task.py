import logging
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm
from dask.distributed import as_completed

from holowizard.core.src.hpc.slurm_cluster import SlurmCluster
from holowizard.core.src.hpc.hpc_serializable import HpcSerializable  

def broadcast_task(
    cluster: SlurmCluster,
    items: Sequence[HpcSerializable],
    task,
    keys: Optional[Sequence[str]] = None,
    from_json_kwargs: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
) -> List[HpcSerializable]:
    """Submit `task(item.to_json())` for every item and deserialize the results.

    Each result is parsed with `type(item).from_json(result, **from_json_kwargs)`,
    so it comes back as the same type as the item that was sent.
    """
    from_json_kwargs = from_json_kwargs or {}

    item_by_key: Dict[str, HpcSerializable] = {item.key: item for item in items}
    if keys is None:
        keys = list(item_by_key)

    missing = [k for k in keys if k not in item_by_key]
    if missing:
        logging.warning(f"No item found for keys: {missing}")

    def submit(key: str):
        future = cluster.submit_task_get_future(task, item_by_key[key].to_json())
        future_to_key[future] = key
        return future

    logging.info("Broadcasting task")
    future_to_key = {}
    retries_by_key = {}
    for key in tqdm(keys):
        if key in item_by_key:
            submit(key)
            retries_by_key[key] = 0

    logging.info("Reading results")
    results: List[HpcSerializable] = []
    pbar = tqdm(total=len(retries_by_key))
    ac = as_completed(list(future_to_key))

    for future in ac:
        key = future_to_key.pop(future)
        try:
            result = future.result()
        except Exception as e:
            retries_by_key[key] += 1
            if retries_by_key[key] <= max_retries:
                logging.warning(
                    f"Task for key {key} failed with {e!r}, retrying "
                    f"({retries_by_key[key]}/{max_retries})"
                )
                ac.add(submit(key))
            else:
                logging.error(f"Task for key {key} failed after {max_retries} retries: {e!r}")
                pbar.update(1)
            continue

        if result is not None:
            item_cls = type(item_by_key[key])
            results.append(item_cls.from_json(str(result), **from_json_kwargs))
        pbar.update(1)

    pbar.close()
    return results