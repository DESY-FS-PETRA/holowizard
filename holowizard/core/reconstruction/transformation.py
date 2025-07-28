import torchvision.transforms as ttf
from torchvision.transforms import InterpolationMode


def resize_guess(oref_predicted, nesterov_vt, new_size, probe_refractive):
    if oref_predicted is not None:
        oref_predicted = (
            ttf.Resize(new_size, interpolation=InterpolationMode.BILINEAR)(oref_predicted.real[None, None, :, :])[
                0, 0, :, :
            ]
            + 1j
            * ttf.Resize(new_size, interpolation=InterpolationMode.BILINEAR)(oref_predicted.imag[None, None, :, :])[
                0, 0, :, :
            ]
        )

    if nesterov_vt is not None:
        nesterov_vt = (
            ttf.Resize(new_size, interpolation=InterpolationMode.BILINEAR)(nesterov_vt.real[None, None, :, :])[
                0, 0, :, :
            ]
            + 1j
            * ttf.Resize(new_size, interpolation=InterpolationMode.BILINEAR)(nesterov_vt.imag[None, None, :, :])[
                0, 0, :, :
            ]
        )

    if probe_refractive is not None:
        probe_refractive = (
            ttf.Resize(new_size, interpolation=InterpolationMode.BILINEAR)(probe_refractive.real[None, None, :, :])[
                0, 0, :, :
            ]
            + 1j
            * ttf.Resize(new_size, interpolation=InterpolationMode.BILINEAR)(probe_refractive.imag[None, None, :, :])[
                0, 0, :, :
            ]
        )

    return oref_predicted, nesterov_vt, probe_refractive
