import traceback


def test_models_convenience_import():
    try:
        from holowizard.core.api.models import ConeBeam

        cone_beam = ConeBeam()
        assert cone_beam is not None

    except Exception:
        print(traceback.format_exc())
        assert False
