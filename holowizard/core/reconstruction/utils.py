from typing import List
import torch
import holowizard.core

if "cuda" in holowizard.core.torch_running_device_name:
    from cupyx.scipy.ndimage import fourier_gaussian
    from cupy import ones
else:
    from numpy import ones
    from scipy.ndimage import fourier_gaussian


def get_filter_kernels(gaussian_filter_fwhm, shape, device):
    if gaussian_filter_fwhm is not None and gaussian_filter_fwhm.real != 0.0:
        filter_kernel_obj_phase = torch.tensor(
            fourier_gaussian(
                ones(shape),
                sigma=gaussian_filter_fwhm.real / 2.35,
            )[:, 0 : int(shape[1] / 2) + 1],
            device=device,
            dtype=torch.float,
        )
    else:
        torch_ones = torch.ones(shape, device=device, dtype=torch.float)
        filter_kernel_obj_phase = torch_ones[:, 0 : int(shape[1] / 2) + 1]

    if gaussian_filter_fwhm is not None and gaussian_filter_fwhm.imag != 0.0:
        filter_kernel_obj_absorption = torch.tensor(
            fourier_gaussian(
                ones(shape),
                sigma=gaussian_filter_fwhm.imag / 2.35,
            )[:, 0 : int(shape[1] / 2) + 1],
            device=device,
            dtype=torch.float,
        )
    else:
        torch_ones = torch.ones(shape, device=device, dtype=torch.float)
        filter_kernel_obj_absorption = torch_ones[:, 0 : int(shape[1] / 2) + 1]

    return filter_kernel_obj_phase, filter_kernel_obj_absorption
