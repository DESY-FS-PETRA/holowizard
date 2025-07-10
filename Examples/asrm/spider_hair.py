import numpy
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import pathlib

from holowizard.core.logging.logger import Logger
from holowizard.core.api.viewer import LossViewer, PyPlotViewer
from holowizard.core.api.base.single_projection.reconstruction import reconstruct
from holowizard.interface.parameters import *
import holowizard.core.utils.fileio as fileio

matplotlib.use("Qt5Agg")

object_shape = (2048, 2048)

root = str(pathlib.Path(__file__).parent.resolve()) + "/"
data_path = root + "../data/holograms/spider_hair.tiff"
working_dir = root + "../logs/"
session_name = "spider_hair_asrm"

Logger.current_log_level = Logger.level_num_image_info
Logger.configure(session_name=session_name, working_dir=working_dir)

flatfield_offset_corr = 0.98
setup = BeamSetup(energy=11.0, px_size=6.5, z02=19.661)
measurements = [
    Measurement(
        data_path=data_path, data=fileio.load_img_data(data_path), z01=7.945
    )
]
padding_options = Padding(
    padding_mode=Padding.PaddingMode.MIRROR_ALL,
    padding_factor=7.0,
    down_sampling_factor=16,
    cutting_band=0,
    a0=flatfield_offset_corr,
)

options_warmup = Options(
    regularization_object=Regularization(
        iterations=700,
        update_rate=0.9,
        l2_weight=0.0 + 10.0 * 1j,
        gaussian_filter_fwhm=2.0 + 0.0j,
    ),
    nesterov_object=Regularization(update_rate=1.0, gaussian_filter_fwhm=16.0 + 16.0j),
    verbose_interval=100,
    padding=deepcopy(padding_options),
)

options_upscale_4 = Options(
    regularization_object=Regularization(
        iterations=300,
        update_rate=1.1,
        l2_weight=0.0 + 1.0 * 1j,
        gaussian_filter_fwhm=2.0 + 8.0j,
    ),
    nesterov_object=Regularization(update_rate=1.0, gaussian_filter_fwhm=8.0 + 8.0j),
    verbose_interval=100,
    padding=deepcopy(padding_options),
)

options_upscale_2 = Options(
    regularization_object=Regularization(
        iterations=500,
        update_rate=1.1,
        l2_weight=0.0 + 0.1 * 1j,
        gaussian_filter_fwhm=2.0 + 8.0j,
    ),
    nesterov_object=Regularization(update_rate=1.0, gaussian_filter_fwhm=64.0 + 64.0j),
    verbose_interval=100,
    padding=deepcopy(padding_options),
    prototype_field=0.0,
)

options_mainrun = Options(
    regularization_object=Regularization(
        iterations=500,
        update_rate=1.1,
        l2_weight=0.0 + 0.0 * 1j,
        gaussian_filter_fwhm=2.0 + 8.0j,
    ),
    nesterov_object=Regularization(update_rate=1.0, gaussian_filter_fwhm=complex(0)),
    verbose_interval=100,
    padding=deepcopy(padding_options),
    prototype_field=0.0,
)

data_dimensions = DataDimensions(
    total_size=object_shape, fov_size=object_shape, window_type="blackman"
)

options_upscale_4.padding.down_sampling_factor = 4
options_upscale_2.padding.down_sampling_factor = 2
options_mainrun.padding.down_sampling_factor = 1

########################################################################################################################
reco_params = RecoParams(
    beam_setup=setup,
    output_path="",
    measurements=measurements,
    reco_options=[
        options_warmup,
        options_upscale_4,
        options_upscale_2,
        options_mainrun,
    ],
    data_dimensions=data_dimensions,
)
result, loss_records = reconstruct(reco_params, viewer=[LossViewer(), PyPlotViewer()])
loss_records = loss_records.cpu()

reco_phaseshift = result.real.cpu().numpy()
reco_absorption = result.imag.cpu().numpy()

plt.close("all")
plt.ioff()

fig, axs = plt.subplots(2, 2)

fig.suptitle(session_name)
img_0 = axs[0, 0].imshow(reco_phaseshift, cmap="gray", interpolation="None")
axs[0, 0].title.set_text("Phaseshift")
plt.colorbar(img_0, orientation="vertical", ax=axs[0, 0])

img_0 = axs[0, 1].imshow(reco_absorption, cmap="gray", interpolation="None")
axs[0, 1].title.set_text("Absorption")
plt.colorbar(img_0, orientation="vertical", ax=axs[0, 1])

axs[1, 0].plot(reco_phaseshift[int(reco_phaseshift.shape[0] / 2), :])
axs[1, 0].title.set_text("Cross section of phases")

axs[1, 1].plot(loss_records)
axs[1, 1].title.set_text("Final MSE Loss: " + str(loss_records[-1]))
axs[1, 1].set_yscale("log")
plt.show()
