import yaml
from omegaconf import OmegaConf

def remove_keys(d, keys_to_remove=("_target_", "_recursive_")):
    if isinstance(d, dict):
        return {
            k: remove_keys(v, keys_to_remove)
            for k, v in d.items() if k not in keys_to_remove
        }
    elif isinstance(d, list):
        return [remove_keys(i, keys_to_remove) for i in d]
    else:
        return d


def remove_hydra_targets(obj):
    """
    Recursively walk a plain Python container and drop any '_target_' keys.
    """
    if isinstance(obj, dict):
        return {
            key: remove_hydra_targets(val)
            for key, val in obj.items()
            if key != "_target_"
        }
    elif isinstance(obj, list):
        return [remove_hydra_targets(x) for x in obj]
    else:
        return obj

def to_clean_yaml(conf):
    # 1) Turn into plain dicts/lists, resolving interpolations
    plain = OmegaConf.to_container(conf, resolve=True)
    # 2) Strip out all '_target_' keys
    cleaned = remove_hydra_targets(plain)
    # 3) Dump to YAML
    return yaml.dump(cleaned, sort_keys=False, default_flow_style=False)