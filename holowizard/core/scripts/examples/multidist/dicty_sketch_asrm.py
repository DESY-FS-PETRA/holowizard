import torch
from copy import deepcopy
import skimage.transform as sktf
import matplotlib
import matplotlib.pyplot as plt
import pathlib

import holowizard.core
from holowizard.core.logging.logger import Logger
from holowizard.core.api.viewer import LossViewer, PyPlotViewer
from holowizard.core.api.functions.single_projection.reconstruction import reconstruct
from holowizard.core.api.models import ConeBeam
from holowizard.core.api.parameters import (
    BeamSetup,
    Measurement,
    Padding,
    Options,
    Regularization,
    DataDimensions,
    RecoParams,
)

if "cuda" in holowizard.core.torch_running_device_name:
    from holowizard.core.models.fresnel_propagator import FresnelPropagator
else:
    from holowizard.core.models.fresnel_propagator_torch import FresnelPropagator

from holowizard.core.phantoms import dicty_sketch
from holowizard.core.preprocessing.process_data_dimensions import (
    process_data_dimensions,
)
from holowizard.core.utils.transform import *

matplotlib.use("Qt5Agg")

object_shape = (2048, 2048)

root = str(pathlib.Path(__file__).parent.resolve()) + "/"
working_dir = root + "../logs/"
session_name = "dicty_sketch_multidist"

Logger.current_log_level = Logger.level_num_image_info
Logger.configure(session_name=session_name, working_dir=working_dir)

setup = BeamSetup(energy=17.0, px_size=0.0065, z02=20_000)
measurements = [Measurement(z01=80.0), Measurement(z01=150.0)]

flatfield_offset_corr = 1.0
padding_options = Padding(
    padding_mode=Padding.PaddingMode.CONSTANT,
    padding_factor=2.0,
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

data_dimensions = DataDimensions(total_size=object_shape, fov_size=object_shape, window_type="blackman")

options_upscale_4.padding.down_sampling_factor = 4
options_upscale_2.padding.down_sampling_factor = 2
options_mainrun.padding.down_sampling_factor = 1

########################################################################################################################

phantom = torch.tensor(
    dicty_sketch(object_shape[0]),
    dtype=torch.cfloat,
    device=holowizard.core.torch_running_device,
)
data_dimensions_processed = process_data_dimensions(deepcopy(data_dimensions), deepcopy(options_mainrun.padding))
model = FresnelPropagator(
    [ConeBeam.get_fr(setup, measurement) for measurement in measurements],
    data_dimensions_processed.total_size,
    holowizard.core.torch_running_device,
)
phantom = pad_to_size(phantom, data_dimensions_processed.total_size)

holograms_sqrt = model.get_measurements_all(phantom)

for i in range(len(measurements)):
    measurements[i].data = crop_center(holograms_sqrt[i], object_shape) ** 2

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

reco_phaseshift = sktf.rotate(result.real.cpu().numpy(), 90)
reco_absorption = sktf.rotate(result.imag.cpu().numpy(), 90)

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

axs[1, 0].plot(reco_phaseshift[1024, :])
axs[1, 0].title.set_text("Cross section of phases")

axs[1, 1].plot(loss_records)
axs[1, 1].title.set_text("Final MSE Loss: " + str(loss_records[-1]))
axs[1, 1].set_yscale("log")
plt.show()
