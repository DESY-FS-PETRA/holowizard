import numpy
from copy import deepcopy
import pathlib

from holowizard.core.logging.logger import Logger
from holowizard.core.api.functions.single_projection.reconstruction import reconstruct
import holowizard.core.utils.fileio as fileio
from holowizard.core.api.parameters import (
    BeamSetup,
    Measurement,
    Padding,
    Options,
    Regularization,
    DataDimensions,
    RecoParams,
)


def test_spider_hair():
    object_shape = (2048, 2048)

    root = str(pathlib.Path(__file__).parent.resolve()) + "/"
    data_path = root + "/data/spider_hair.tiff"
    working_dir = root + "/logs/"
    session_name = "spider_hair_asrm"

    Logger.current_log_level = Logger.level_num_image_debug
    Logger.configure(session_name=session_name, working_dir=working_dir)

    flatfield_offset_corr = 0.98
    setup = BeamSetup(energy=11.0, px_size=6.5, z02=19.661)
    measurements = [Measurement(data_path=data_path, data=fileio.load_img_data(data_path), z01=7.945)]
    padding_options = Padding(
        padding_mode=Padding.PaddingMode.MIRROR_ALL,
        padding_factor=4.0,
        down_sampling_factor=16,
        cutting_band=0,
        a0=flatfield_offset_corr,
    )

    options_warmup = Options(
        regularization_object=Regularization(
            iterations=200,
            update_rate=0.9,
            l2_weight=0.0 + 10.0 * 1j,
            gaussian_filter_fwhm=2.0 + 0.0j,
        ),
        nesterov_object=Regularization(update_rate=1.0, gaussian_filter_fwhm=16.0 + 16.0j),
        verbose_interval=100,
        padding=deepcopy(padding_options),
    )

    options_mainrun = Options(
        regularization_object=Regularization(
            iterations=200,
            update_rate=0.9,
            l2_weight=0.0 + 0.0 * 1j,
            gaussian_filter_fwhm=2.0 + 8.0j,
        ),
        nesterov_object=Regularization(update_rate=0.5, gaussian_filter_fwhm=complex(0)),
        verbose_interval=100,
        padding=deepcopy(padding_options),
        prototype_field=0.0,
    )

    data_dimensions = DataDimensions(total_size=object_shape, fov_size=object_shape, window_type="blackman")

    options_mainrun.padding.down_sampling_factor = 1

    ########################################################################################################################
    reco_params = RecoParams(
        beam_setup=setup,
        output_path="",
        measurements=measurements,
        reco_options=[options_warmup, options_mainrun],
        data_dimensions=data_dimensions,
    )
    result, _ = reconstruct(reco_params, viewer=[])

    reco_phaseshift = result.real.cpu().numpy()
    reco_absorption = result.imag.cpu().numpy()
    reference_absorption = fileio.load_img_data(root + "reference_results/absorption.tiff")
    reference_phaseshift = fileio.load_img_data(root + "reference_results/phaseshift.tiff")
    print(
        "norm-difference phaseshift: ",
        numpy.linalg.norm(reco_phaseshift - reference_phaseshift),
    )
    print(
        "norm-difference absorption: ",
        numpy.linalg.norm(reco_absorption - reference_absorption),
    )
    # assert numpy.allclose(reco_phaseshift, reference_phaseshift, rtol=1e-8)
    # assert numpy.allclose(reco_absorption, reference_absorption, rtol=1e-8)
    assert not numpy.isnan(reco_phaseshift).any()
    assert not numpy.isnan(reco_absorption).any()
