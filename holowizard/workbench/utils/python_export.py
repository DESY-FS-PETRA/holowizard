"""Python script of a test phase retrieval ("Download Python" on the phase retrieval page).

The script repeats what "Test Phase Retrieval" does for one hologram, with the same parameters: flatfield correction
with the PCA components of the reference images (computed first if the file does not exist yet), then the phase
retrieval stages. All values are written out explicitly, so the script runs without the workbench and its config.
"""

import datetime
import math
import sys
from typing import Optional

from holowizard.core.api.parameters import Options, Padding, RecoParams, Regularization

INDENT = "    "


def _float(x: float) -> str:
    """A float literal; +-sys.float_info.max symbolically (the "no limit" values of the regularization)."""
    x = float(x)
    if x == sys.float_info.max:
        return "sys.float_info.max"
    if x == -sys.float_info.max:
        return "-sys.float_info.max"
    return repr(x)


def _complex(c: complex, imag_text: Optional[str] = None) -> str:
    c = complex(c)
    return f"complex({_float(c.real)}, {imag_text or _float(c.imag)})"


def _value(v) -> str:
    if isinstance(v, complex):
        return _complex(v)
    if isinstance(v, float):
        return _float(v)
    return repr(v)


def _regularization(r: Regularization, a0: float) -> str:
    # values_min "auto" is -inf + i*log(a0) (see build_reco_params): written with a0 so that it follows a change of a0.
    values_min = complex(r.values_min)
    log_a0 = "np.log(a0)" if values_min.imag == math.log(a0) and values_min.imag != 0 else None
    args = [
        f"iterations={_value(r.iterations)}",
        f"update_rate={_value(r.update_rate)}",
        f"gaussian_filter_fwhm={_complex(r.gaussian_filter_fwhm)}",
        f"values_min={_complex(values_min, log_a0)}",
        f"values_max={_complex(r.values_max)}",
        f"l2_weight={_complex(r.l2_weight)}",
        f"l1_weight={_complex(r.l1_weight)}",
    ]
    i3 = INDENT * 3
    return "Regularization(\n" + "".join(f"{i3}{arg},\n" for arg in args) + f"{INDENT * 2})"


def _padding(p: Padding, a0: float) -> str:
    args = [
        f"padding_mode=Padding.PaddingMode.{p.padding_mode.name}",
        f"padding_factor={_value(p.padding_factor)}",
        f"down_sampling_factor={_value(p.down_sampling_factor)}",
        f"cutting_band={_value(p.cutting_band)}",
        "a0=a0" if float(p.a0) == float(a0) else f"a0={_value(p.a0)}",
        f"prototype_field={_value(p.prototype_field)}",
    ]
    return "Padding(" + ", ".join(args) + ")"


def _options(o: Options, a0: float) -> str:
    i2 = INDENT * 2
    lines = [
        f"{i2}update_blocks={_value(o.update_blocks)},",
        f"{i2}regularization_object={_regularization(o.regularization_object, a0)},",
        f"{i2}nesterov_object={_regularization(o.nesterov_object, a0)},",
        f"{i2}regularization_probe={_regularization(o.regularization_probe, a0)},",
        f"{i2}z01_tol={_value(o.z01_tol)},",
        f"{i2}padding={_padding(o.padding, a0)},",
        f"{i2}verbose_interval={_value(o.verbose_interval)},",
        f"{i2}prototype_field={_value(o.prototype_field)},",
    ]
    return f"{INDENT}Options(\n" + "\n".join(lines) + f"\n{INDENT}),"


def test_phase_retrieval_script(
    reco_params: RecoParams,
    a0: float,
    scan_name: str,
    hologram_file: str,
    reference_dir: str,
    reference_key: str,
    components_path: str,
    num_components: int,
    tomogram: Optional[str] = None,
    created_by: Optional[str] = None,
) -> str:
    """Source of a standalone script that runs the test phase retrieval of ``hologram_file`` with ``reco_params``
    (as built by ``build_reco_params`` for the test phase retrieval on the phase retrieval page)."""
    beam = reco_params.beam_setup
    measurement = reco_params.measurements[0]
    dims = reco_params.data_dimensions
    stages = "\n".join(_options(o, a0) for o in reco_params.reco_options)
    created = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    source = f"{scan_name}/{tomogram}" if tomogram else scan_name
    return f'''"""Test phase retrieval of one hologram, as "Test Phase Retrieval" on the phase retrieval page of the
HoloWizard workbench (exported {created}{f" by {created_by}" if created_by else ""}).

Scan: {source}, hologram: {hologram_file.rsplit("/", 1)[-1]}

Steps: flatfield correction with the PCA components of the reference images (computed first if the components file
does not exist yet), then the phase retrieval stages below. The result is shown and saved next to this script.
"""

import datetime
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skio

from holowizard.core.api.functions.flatfield_correction.calc_flatfield_components_o import (
    calculate_flatfield_components,
)
from holowizard.core.api.functions.single_projection.reconstruction_flatfieldcorrection import reconstruct
from holowizard.core.api.parameters import (
    BeamSetup,
    DataDimensions,
    FlatfieldComponentsParams,
    FlatfieldCorrectionParams,
    Measurement,
    Options,
    Padding,
    RecoParams,
    Regularization,
)
from holowizard.core.api.viewer import LossViewer, PyPlotViewer
from holowizard.core.src.logging.logger import Logger
from holowizard.core.src.misc.fileio import write_img_data
from holowizard.core.src.misc.transform import crop_center

# --- Data ---
hologram_file = {hologram_file!r}
# reference images: the files of this folder with "{reference_key}" in the name, as the workbench selects them
reference_dir = Path({reference_dir!r})
reference_files = sorted(str(p) for p in reference_dir.iterdir() if {reference_key!r} in p.name)
# PCA components of the reference images (the file the workbench uses; computed if it does not exist)
components_path = {components_path!r}
num_components = {num_components!r}
output_dir = Path(__file__).resolve().parent / "test_phase_retrieval_output"

# --- Geometry (mm, keV) ---
energy = {_float(beam.energy)}
px_size = {_float(beam.px_size)}  # detector pixel size
z01 = {_float(measurement.z01)}
z02 = {_float(beam.z02)}
z01_confidence = {_value(measurement.z01_confidence)}
a0 = {_float(a0)}

# --- Phase retrieval stages (parameter form of the phase retrieval page) ---
stages = [
{stages}
]

data_dimensions = DataDimensions(
    total_size={tuple(dims.total_size)!r},
    fov_size={tuple(dims.fov_size)!r},
    window_type={dims.window_type!r},
)

########################################################################################################################
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
Logger.current_log_level = Logger.level_num_loss
Logger.configure(session_name=f"{{timestamp}}_test_phase_retrieval", working_dir=str(output_dir / "log"))

if not Path(components_path).is_file():
    print(f"Calculating the flatfield components from {{len(reference_files)}} reference images: {{components_path}}")
    Path(components_path).parent.mkdir(parents=True, exist_ok=True)
    calculate_flatfield_components(
        FlatfieldComponentsParams(
            measurements=[skio.imread(f) for f in reference_files],
            num_components=num_components,
            save_path=components_path,
        )
    )

reco_params = RecoParams(
    beam_setup=BeamSetup(energy=energy, px_size=px_size, z02=z02),
    measurements=[Measurement(z01=z01, z01_confidence=z01_confidence, data=skio.imread(hologram_file))],
    reco_options=stages,
    data_dimensions=data_dimensions,
    output_path="",
)

result, se_losses = reconstruct(
    flatfield_correction_params=FlatfieldCorrectionParams(image=hologram_file, components_path=components_path),
    reco_params=reco_params,
    viewer=[LossViewer(), PyPlotViewer()],
)

phaseshift = crop_center(np.float32(np.real(result.cpu().numpy())), data_dimensions.fov_size)
absorption = crop_center(np.float32(np.imag(result.cpu().numpy())), data_dimensions.fov_size)
name = Path(hologram_file).stem
write_img_data(str(output_dir / f"phase_{{name}}.tiff"), phaseshift)
write_img_data(str(output_dir / f"absorption_{{name}}.tiff"), absorption)
print(f"Saved phase_{{name}}.tiff and absorption_{{name}}.tiff in {{output_dir}}")

# --- Result ---
losses = se_losses.cpu().numpy()
plt.close("all")
plt.ioff()
fig, axs = plt.subplots(2, 2, figsize=(11, 9))
fig.suptitle(f"{source} / {{name}}")
img = axs[0, 0].imshow(phaseshift, cmap="gray", interpolation="none")
axs[0, 0].set_title("Phaseshift")
plt.colorbar(img, ax=axs[0, 0])
img = axs[0, 1].imshow(absorption, cmap="gray", interpolation="none")
axs[0, 1].set_title("Absorption")
plt.colorbar(img, ax=axs[0, 1])
axs[1, 0].plot(phaseshift[phaseshift.shape[0] // 2, :])
axs[1, 0].set_title("Cross section of the phaseshift")
axs[1, 1].plot(losses)
axs[1, 1].set_yscale("log")
axs[1, 1].set_title(f"Final loss: {{losses[-1]:.4g}}")
fig.tight_layout()
fig.savefig(output_dir / f"result_{{name}}.png", dpi=120)
plt.show()
'''
