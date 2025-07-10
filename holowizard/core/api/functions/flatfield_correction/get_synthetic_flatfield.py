import pickle
from holowizard.core.preprocessing.correct_flatfield import (
    get_synthetic_flatfield as get_synthetic_flatfield_internal,
)

from holowizard.core.parameters.flatfield_correction_params import FlatfieldCorrectionParams


def get_synthetic_flatfield(flatfield_correction_params: FlatfieldCorrectionParams):
    with open(flatfield_correction_params.components_path, "rb") as file:
        components_model = pickle.load(file)

    return get_synthetic_flatfield_internal(
        flatfield_correction_params.image, components_model
    )
