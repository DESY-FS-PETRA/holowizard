import numpy as np
from holowizard.interface.parameters import *
from holowizard.interface.network.send import send


def reconfigure_logger(working_dir, session_name):
    return send(
        "reconfigure_logger", working_dir=working_dir, session_name=session_name
    )


def reconstruct(
    flatfield_correction_params: FlatfieldCorrectionParams, reco_params: RecoParams
):
    return send(
        "reconstruct",
        flatfield_correction_params=flatfield_correction_params,
        reco_params=reco_params,
    )


def add_flatfield(measurement: np.ndarray):
    return send("add_flatfield", measurement=measurement)


def reset_flatfield_list():
    return send("reset_flatfield_list")


def correct_flatfield(flatfield_correction_params: FlatfieldCorrectionParams):
    return send(
        "correct_flatfield", flatfield_correction_params=flatfield_correction_params
    )


def calculate_flatfield_components(
    flatfield_components_params: FlatfieldComponentsParams,
):
    return send(
        "calculate_flatfield_components",
        flatfield_components_params=flatfield_components_params,
    )


def find_focus(
    flatfield_correction_params: FlatfieldCorrectionParams, reco_params: RecoParams
):
    return send(
        "find_focus",
        flatfield_correction_params=flatfield_correction_params,
        reco_params=reco_params,
    )
