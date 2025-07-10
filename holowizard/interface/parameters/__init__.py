import sys

member_value_adapter = None

try:
    import torch
except Exception:
    pass

if "torch" in sys.modules:
    from holowizard.interface.type_conversion.member_value_adapter_torch import (
        MemberValueAdapterTorch,
    )

    member_value_adapter = MemberValueAdapterTorch
else:
    from holowizard.interface.type_conversion.member_value_adapter_numpy import (
        MemberValueAdapterNumpy,
    )

    member_value_adapter = MemberValueAdapterNumpy

from holowizard.interface.parameters.measurement import Measurement
from holowizard.interface.parameters.beam_setup import BeamSetup
from holowizard.interface.parameters.data_dimensions import DataDimensions
from holowizard.interface.parameters.flatfield_components_params import (
    FlatfieldComponentsParams,
)
from holowizard.interface.parameters.flatfield_correction_params import (
    FlatfieldCorrectionParams,
)
from holowizard.interface.parameters.measurement import Measurement
from holowizard.interface.parameters.options import Options
from holowizard.interface.parameters.padding import Padding
from holowizard.interface.parameters.reco_params import RecoParams
from holowizard.interface.parameters.regularization import Regularization
from holowizard.interface.parameters.dask_options import DaskOptions
