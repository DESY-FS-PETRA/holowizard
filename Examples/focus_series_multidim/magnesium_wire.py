import numpy
import os
from copy import deepcopy
from skimage import io
import skimage.transform as sktf

from holowizard.core.logging.logger import Logger
from holowizard.core.parameters import *
from holowizard.core.api.parameters.paths.focus_series_paths import FocusSeriesPaths
from holowizard.core.api.base.single_projection.reconstruction import reconstruct
from holowizard.core.utils.fileio import load_img_data
from holowizard.core.api.parameters.paths.project_paths import ProjectPaths

z01_guess = 48.051

z01_resolution = int(sys.argv[1])
z01_slurm_job_id_int = int(sys.argv[2])
z01_slurm_job_id = str(z01_slurm_job_id_int).zfill(4)
z01_confidence = 3.0
z01_confidence_interval = numpy.linspace(
    -z01_confidence, z01_confidence, z01_resolution
)
z01_current_corr = z01_confidence_interval[int(z01_slurm_job_id_int % z01_resolution)]

a0_resolution = int(sys.argv[3])
a0_slurm_job_id_int = int(sys.argv[4])
a0_slurm_job_id = str(a0_slurm_job_id_int).zfill(4)
a0_range = (0.3, 1.8)
a0_confidence_interval = numpy.linspace(a0_range[0], a0_range[1], a0_resolution)
a0_current = a0_confidence_interval[int(a0_slurm_job_id_int % a0_resolution)]

focus_series_paths = FocusSeriesPaths(
    root_dir="/gpfs/petra3/scratch/"
    + os.environ.get("USER")
    + "/focus_series_multidim/magnesium_wire"
)

project_paths = ProjectPaths(
    output_dir="/gpfs/petra3/scratch/"
    + os.environ.get("USER")
    + "/focus_series_multidim/magnesium_wire/projections",
    session_name="magnesium_wire_focus_series",
    session_id=z01_slurm_job_id + "_" + a0_slurm_job_id,
    other=focus_series_paths,
)

project_paths.data_path = (
    os.path.dirname(os.path.realpath(__file__))
    + "/../data/holograms/magnesium_wire.tiff"
)
project_paths.logs_dir = os.path.dirname(os.path.realpath(__file__)) + "/../logs"

Logger.current_log_level = Logger.level_num_loss
Logger.configure(
    session_name=project_paths.session_logs_name, working_dir=project_paths.logs_dir
)
project_paths.mkdirs()

flatfield_offset_corr = a0_current
setup = BeamSetup(energy=11.0, px_size=6.5, z02=19.661)
measurements = [
    Measurement(
        data_path=project_paths.data_path,
        data=load_img_data(project_paths.data_path),
        z01=z01_guess + z01_current_corr,
        z01_confidence=z01_confidence,
    )
]
padding_options = Padding(
    padding_mode=Padding.PaddingMode.MIRROR_ALL,
    padding_factor=4,
    down_sampling_factor=16,
    cutting_band=0,
    a0=a0_current,
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

result, loss_records = reconstruct(reco_params)

x_predicted = result.cpu().numpy()
loss_records = loss_records.cpu().numpy()

reco_phaseshift = sktf.rotate(x_predicted.real, 90)

with open(
    focus_series_paths.se_losses
    + "/loss_"
    + str(z01_current_corr)
    + "_"
    + str(a0_current)
    + ".txt",
    "w",
) as f:
    f.write(
        "("
        + str(z01_current_corr)
        + ","
        + str(a0_current)
        + ","
        + str(loss_records[-1])
        + ")"
    )

io.imsave(
    focus_series_paths.projections_dir
    + "/img_"
    + z01_slurm_job_id
    + "_"
    + a0_slurm_job_id
    + "_"
    + str(z01_current_corr).zfill(2)
    + "_"
    + str(a0_current).zfill(2)
    + ".tiff",
    reco_phaseshift,
)
