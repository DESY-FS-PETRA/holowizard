from holowizard.core.api.parameters import (
    RecoParams,
    BeamSetup,
    Measurement,
    DataDimensions,
)
from holowizard.pipe.scan import Scan
from hydra.utils import instantiate
import sys
import numpy as np


def build_reco_params(scan: Scan, reco_params) -> RecoParams:
    """
    Build a RecoParams object from the configuration dictionary for Find Focus.

    Args:
        config (dict): Parsed holopipe configuration.

    Returns:
        RecoParams: The constructed runtime object.
    """
    cfg = scan.config

    beam_setup = BeamSetup(
        energy=scan.energy,
        px_size=cfg.scan.px_size,
        z02=scan.z02,  # Use the z02 from the scan object
    )

    measurements = [
        Measurement(
            data_path="",
            z01=scan.z01,  # Use the z01 from the scan object
            z01_confidence=cfg.scan.z01_confidence,  # Use the z01_confidence from the scan object)
        )
    ]

    reco_options_find_focus = [
        instantiate(
            o,
            regularization_object={
                "values_min": (
                    -sys.float_info.max + 1j * np.log(scan.a0)
                    if o.regularization_object.get("values_min", "auto") == "auto"
                    else o.regularization_object.values_min
                ),
                "values_max": (
                    0.0 + sys.float_info.max * 1j
                    if o.regularization_object.get("values_max", "auto") == "auto"
                    else o.regularization_object.values_max
                ),
            },
            padding={"a0": float(scan.a0), "padding_mode": "MIRROR_ALL"},
        )
        for o in reco_params.stages
    ]

    data_dimensions = DataDimensions(
        total_size=tuple(cfg.scan.data.total_size),
        fov_size=tuple(cfg.scan.data.fov_size),
        window_type=cfg.scan.data.window_type,
    )

    return RecoParams(
        beam_setup=beam_setup,
        output_path="",  # to be set in processing step
        measurements=measurements,
        reco_options=reco_options_find_focus,
        data_dimensions=data_dimensions,
    )
