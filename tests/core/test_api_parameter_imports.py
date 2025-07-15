import traceback

def test_parameters_convenience_import():
    try:
        from holowizard.core.api.parameters import BeamSetup, Measurement, Padding, Options, Regularization, DataDimensions, RecoParams

        beam_setup = BeamSetup(17,0.65,20_00_0)
        measurement = Measurement(1)
        padding = Padding()
        options = Options()
        regularization = Regularization()
        data_dimensions = DataDimensions((100,100),(100,100),"hanning")

        assert beam_setup is not None
        assert measurement is not None
        assert padding is not None
        assert options is not None
        assert regularization is not None
        assert data_dimensions is not None

        reco_params = RecoParams(beam_setup, [measurement], [options], data_dimensions, "")
        assert reco_params is not None

    except Exception:
        print(traceback.format_exc())
        assert False