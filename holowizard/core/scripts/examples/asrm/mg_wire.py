###### General imports
from copy import deepcopy
import numpy
import torch
import glob
import tifffile
import numpy as np

##### Holowizard imports for 2D ASRM

from holowizard.core.logging.logger import Logger
from holowizard.core.api.viewer import LossViewer, PyPlotViewer, ZeroMQViewer
from holowizard.core.api.parameters import (
    BeamSetup,
    Measurement,
    Padding,
    DaskOptions,
    Regularization,
    DataDimensions,
    RecoParams,
)


###### Holowizard imports for reprojection routine

from holowizard.core.reconstruction.methods.reprojection.repro_options import ReproOptions
from holowizard.core.reconstruction.methods.reprojection.reprojection_reconstruction import reconstruct as reprojection_reconstruction

########################################################################################################################################################

output_path = "/gpfs/petra3/scratch/dorajoha/envs/holowizard_main/logs/"

#downsampling angles for debugging....
downsample_angles_debugging = 16

#### Load angles

filename = "/asap3/petra3/gpfs/p05/2020/data/11008588/raw/nano3674_Mg-6Ag_r_3day_3/nano3674_Mg-6Ag_r_3day_3__LogScan.log"

lines = [line.rstrip('\n') for line in open(filename)]

lines = lines[10:]
angles = []
img_str = []
for i, line in enumerate(lines):
    line2 = lines[i].split()[8]
    line3 = lines[i].split()[0]
    if "img_y0" in line3:
        angles.append(np.round(float(line2), 3))     #* np.pi / 180)
        img_str.append(str(line3))

angles = torch.tensor(angles)

### DOWNSAMPLING FOR DEBUGGING CT...

### for some reason we need the minus for the mg wire. achja. idk
angles = -angles[::downsample_angles_debugging]

##### Load corrected holograms

object_shape = (2048, 2048)
#root = str(pathlib.Path(__file__).parent.resolve()) + "/"

root = f"/data/dust/user/hernande/holow_repro_scripts/data/mg_holograms/"
image_file_list = glob.glob(root+"img_*.tiff")
image_file_list.sort()


### DOWNSAMPLING FOR DEBUGGING CT...
image_file_list = image_file_list[::downsample_angles_debugging]

#print("Exists:", os.path.isdir(root))
#print("Files:", os.listdir(root))

image_data_list = [tifffile.imread(image_name) for image_name in image_file_list]
#image_data_list = [np.rot90(tifffile.imread(image_name), k=2).copy() for image_name in image_file_list]
#image_data_list = image_data_list[:-1]

print(len(angles))
print(len(image_data_list))


##### Set up logger

working_dir = output_path
session_name = "mg_wire_repro"

Logger.current_log_level = Logger.level_num_loss
Logger.configure(session_name=session_name, working_dir=working_dir)

##### Set up measurements

z01 = 470.71
measurements = [Measurement(angle=i,data_path="", data=torch.tensor(image_data_list[i],device=torch.device("cpu"),dtype=torch.float), z01=z01) for i in range(len(image_data_list))]

##### Set up single projection single stage ASRM options together with tomography and angle selection options per stage (RepoOptions(Options))

flatfield_offset_corr = 0.8
setup = BeamSetup(energy=11, px_size=0.0065, z02=19_661.0)

padding_options = Padding(
    padding_mode=Padding.PaddingMode.MIRROR_ALL,
    padding_factor=4,
    down_sampling_factor=16,
    cutting_band=0,
    a0=flatfield_offset_corr,
)

options_warmup = ReproOptions(
    ct_alg =  "SIRT",    #### wont be used, if update_blocks = 1 (hardcoded FBP to be used first always!)
    ct_params = "10",    #### wont be used, if update_blocks = 1 (hardcoded FBP to be used first always!)
    as_alg = "white",
    #as_params =   "69", 
    as_params= str(len(angles)),
    angles =  angles,
    update_blocks = 1,
    regularization_object=Regularization(
        iterations=700,
        update_rate=0.9,
        l2_weight=0.0 + 10.0 * 1j,
        gaussian_filter_fwhm=2.0 + 0.0j,
    ),
    nesterov_object=Regularization(update_rate=1.0, gaussian_filter_fwhm=4.0 + 4.0j),
    verbose_interval=100,
    padding=deepcopy(padding_options),
)

options_upscale_8 = ReproOptions(
    #ct_alg =  "TV_SIRT",
    #ct_params = [100,0.0001],    
    ct_alg =  "SIRT",
    ct_params = "100",    
    as_alg = "white",
    as_params =   "20", 
    angles =  angles,
    update_blocks = 5,
    regularization_object=Regularization(
        iterations=200,
        update_rate=0.9,
        l2_weight=0.0 + 10.0 * 1j,
        gaussian_filter_fwhm=2.0 + 0.0j,
    ),
    nesterov_object=Regularization(update_rate=1.0, gaussian_filter_fwhm=4.0 + 4.0j),
    verbose_interval=100,
    padding=deepcopy(padding_options),
)


options_upscale_4 = ReproOptions(
    ct_alg =  "SIRT", 
    ct_params = "10", 
    as_alg = "white",
    as_params =   "20", 
    angles =  angles,
    update_blocks = 5,
    regularization_object=Regularization(
        iterations=200,
        update_rate=1.1,
        l2_weight=0.0 + 1.0 * 1j,
        gaussian_filter_fwhm=2.0 + 8.0j,
    ),
    nesterov_object=Regularization(update_rate=0.5, gaussian_filter_fwhm=8.0 + 8.0j),
    verbose_interval=100,
    padding=deepcopy(padding_options),
)

options_upscale_2 = ReproOptions(
    ct_alg =  "SIRT", 
    ct_params = "10", 
    as_alg = "white",
    as_params =   "20", 
    angles =  angles,
    update_blocks = 1,
    regularization_object=Regularization(
        iterations=200,
        update_rate=1.1,
        l2_weight=0.0 + 0.1 * 1j,
        gaussian_filter_fwhm=2.0 + 8.0j,
    ),
    nesterov_object=Regularization(update_rate=0.5, gaussian_filter_fwhm=32.0 + 32.0j),
    verbose_interval=100,
    padding=deepcopy(padding_options),
    prototype_field=0.0,
)

options_mainrun = ReproOptions(
    ct_alg =  "SIRT", 
    ct_params = "5", 
    as_alg = "white",
    as_params =   "20", 
    angles =  angles,
    update_blocks = 1,
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

data_dimensions = DataDimensions(total_size=(2048, 2048), fov_size=(2048, 2048), window_type="blackman")

options_upscale_8.padding.down_sampling_factor = 16
options_upscale_4.padding.down_sampling_factor = 8
options_upscale_2.padding.down_sampling_factor = 2
options_mainrun.padding.down_sampling_factor = 1

#### Set up reco_params

reco_params = RecoParams(
    beam_setup=setup,
    output_path=output_path,
    measurements=measurements,
    reco_options=[options_warmup, options_upscale_8, options_upscale_4, options_upscale_2, options_mainrun],
    data_dimensions=data_dimensions
)
###### Reconstruct

dask_options = DaskOptions(working_dir=working_dir + "/dask_worker/",
                           partitions="maxgpu,allgpu",
                           num_worker=numpy.minimum(len(measurements),80),
                           python_env="/gpfs/petra3/scratch/dorajoha/envs/holowizard_main_env",
                           constraint="A100|V100")

result = reprojection_reconstruction(reco_params, viewer=[LossViewer()], dask_options=dask_options)

