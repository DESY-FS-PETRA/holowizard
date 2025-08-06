import traceback
import torch


def test_models_convenience_import():
    try:
        from holowizard.core.api.models import ConeBeam
        from holowizard.core.api.models import FresnelPropagator

        cone_beam = ConeBeam()
        fresnel_propagator = FresnelPropagator([0.001], (2048, 2048), torch.device("cpu"))

        assert cone_beam is not None
        assert fresnel_propagator is not None

    except Exception:
        print(traceback.format_exc())
        assert False
