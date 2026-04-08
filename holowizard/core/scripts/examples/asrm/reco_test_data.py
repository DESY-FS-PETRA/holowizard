from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import pathlib
import torch
import h5py

import holowizard.core
from holowizard.core.logging.logger import Logger
from holowizard.core.api.viewer import LossViewer, PyPlotViewer
from holowizard.core.api.functions.single_projection.reconstruction import reconstruct
from holowizard.core.api.parameters import (
    BeamSetup,
    Measurement,
    Padding,
    Options,
    Regularization,
    DataDimensions,
    RecoParams,
)
import holowizard.core.utils.fileio as fileio

matplotlib.use("Qt5Agg")

test_file = "/gpfs/petra3/scratch/dorajoha/envs/temp/train.hdf5"
holo_index = 8

with h5py.File(test_file, 'r') as f:
    hologram = f["/images/hologram"][holo_index,:,:]
    z01 = float(f["/metadata/setup/z01"][holo_index])
    z02 = float(f["/metadata/setup/z02"][holo_index])
    energy = float(f["/metadata/setup/energy"][holo_index])
    detector_px_size = float(f["/metadata/setup/detector_px_size"][holo_index])


root = str(pathlib.Path(__file__).parent.resolve()) + "/"

working_dir = root + "../logs/"
session_name = "test_data"

Logger.current_log_level = Logger.level_num_image_debug
Logger.configure(session_name=session_name, working_dir=working_dir)

flatfield_offset_corr = 1.0
setup = BeamSetup(energy=energy, px_size=detector_px_size, z02=z02)
measurements = [Measurement(data_path="", data=hologram, z01=z01)]
padding_options = Padding(
    padding_mode=Padding.PaddingMode.MIRROR_ALL,
    padding_factor=4,
    down_sampling_factor=4,
    cutting_band=0,
    a0=flatfield_offset_corr,
)

options_warmup = Options(
    regularization_object=Regularization(
        iterations=700,
        update_rate=0.9,
        l2_weight=0.0 + 1.0 * 1j,
        gaussian_filter_fwhm=0.5 + 8.0j,
    ),
    nesterov_object=Regularization(update_rate=1.0, gaussian_filter_fwhm=8.0 + 8.0j),
    verbose_interval=100,
    padding=deepcopy(padding_options),
)

options_warmup_2 = Options(
    regularization_object=Regularization(
        iterations=500,
        update_rate=1.1,
        l2_weight=0.0 + 1.0 * 1j,
        gaussian_filter_fwhm=0.5 + 8.0j,
    ),
    nesterov_object=Regularization(update_rate=1.0, gaussian_filter_fwhm=8.0 + 8.0j),
    verbose_interval=100,
    padding=deepcopy(padding_options),
)

options_mainrun = Options(
    regularization_object=Regularization(
        iterations=2000,
        update_rate=0.9,
        l2_weight=0.0 + 0.0 * 1j,
        gaussian_filter_fwhm=0.5 + 8.0j,
    ),
    nesterov_object=Regularization(update_rate=0.5, gaussian_filter_fwhm=0.0 + 0.0j),
    verbose_interval=100,
    padding=deepcopy(padding_options),
    prototype_field=0.0,
)

data_dimensions = DataDimensions(total_size=hologram.shape, fov_size=hologram.shape, window_type="blackman")

options_warmup_2.padding.down_sampling_factor = 2
options_mainrun.padding.down_sampling_factor = 1

########################################################################################################################
reco_params = RecoParams(
    beam_setup=setup,
    output_path="",
    measurements=measurements,
    reco_options=[
        options_warmup,
        options_warmup_2,
        options_mainrun,
    ],
    data_dimensions=data_dimensions
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
