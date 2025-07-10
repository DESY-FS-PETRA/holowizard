import os
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import pathlib

from holowizard.core.logging.logger import Logger
from holowizard.core.api.viewer import LossViewer, PyPlotViewer
from holowizard.core.api.base.find_focus.find_focus import find_focus
from holowizard.core.utils.fileio import load_img_data
from holowizard.core.api.parameters.paths.project_paths import ProjectPaths

matplotlib.use("Qt5Agg")

z01_guess = 7.995
z01_confidence = 1.0

project_paths = ProjectPaths(
    root_dir=str(pathlib.Path(__file__).parent.resolve()) + "/",
    session_name="spider_hair_find_focus",
    session_id=0,
)

project_paths.data_path = (
    os.path.dirname(os.path.realpath(__file__)) + "/../data/holograms/spider_hair.tiff"
)
project_paths.logs_dir = os.path.dirname(os.path.realpath(__file__)) + "/../logs"

Logger.current_log_level = Logger.level_num_loss
Logger.configure(
    session_name=project_paths.session_logs_name, working_dir=project_paths.logs_dir
)

flatfield_offset_corr = 0.98
setup = BeamSetup(energy=11.0, px_size=6.5, z02=19.661)
measurements = [
    Measurement(
        data_path=project_paths.data_path,
        data=load_img_data(project_paths.data_path),
        z01=z01_guess,
        z01_confidence=z01_confidence,
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
    nesterov_object=Regularization(update_rate=1.0, gaussian_filter_fwhm=8.0 + 8.0j),
    verbose_interval=100,
    padding=deepcopy(padding_options),
)

options_upscale_4 = Options(
    regularization_object=Regularization(
        iterations=300,
        update_rate=1.1,
        l2_weight=0.0 + 10.0 * 1j,
        gaussian_filter_fwhm=2.0 + 8.0j,
    ),
    nesterov_object=Regularization(update_rate=1.0, gaussian_filter_fwhm=16.0 + 16.0j),
    verbose_interval=100,
    padding=deepcopy(padding_options),
)

options_upscale_4_lowreg = Options(
    regularization_object=Regularization(
        iterations=500,
        update_rate=1.1,
        l2_weight=0.0 + 1.0 * 1j,
        gaussian_filter_fwhm=2.0 + 8.0j,
    ),
    nesterov_object=Regularization(update_rate=1.0, gaussian_filter_fwhm=16.0 + 16.0j),
    verbose_interval=100,
    padding=deepcopy(padding_options),
)

data_dimensions = DataDimensions(
    total_size=(2048, 2048), fov_size=(2048, 2048), window_type="blackman"
)

options_upscale_4.padding.down_sampling_factor = 4
options_upscale_4_lowreg.padding.down_sampling_factor = 4

reco_params = RecoParams(
    beam_setup=setup,
    output_path=project_paths.output_dir,
    measurements=measurements,
    reco_options=[options_warmup, options_upscale_4, options_upscale_4_lowreg],
    data_dimensions=data_dimensions,
)

result, z01_records, loss_values_history = find_focus(
    reco_params, viewer=[LossViewer(), PyPlotViewer()]
)

print("Found z01=", result, " after ", len(z01_records), " iterations")

plt.close("all")
plt.ioff()

plt.plot(z01_records, loss_values_history)

ylims = plt.ylim()
ylims = (ylims[0] - (ylims[1] - ylims[0]) / 6, ylims[1])
plt.ylim(ylims)

for i in range(4):
    plt.annotate(
        str(i),
        (z01_records[i], loss_values_history[i]),
        xytext=(0, -20),
        textcoords="offset points",
        size="large",
        horizontalalignment="center",
    )

plt.show()

#### For additional a0 search
"""
z01_guess, z01_records, a0_guess, a0_records, loss_values_history = find_focus(reco_params,viewer=[LossViewer()])

print("Found z01=",z01_guess,"a0=",a0_guess," after ", len(loss_values_history)," iterations")

plt.close("all")
plt.ioff()

ax = plt.figure().add_subplot(projection='3d')
ax.plot(z01_records,a0_records,loss_values_history)

plt.show()
"""
