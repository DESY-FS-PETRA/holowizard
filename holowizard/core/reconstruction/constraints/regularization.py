import torch

from holowizard.core.parameters.padding import Padding
from holowizard.core.parameters.data_dimensions import DataDimensions
from holowizard.core.preprocessing.process_image import flip_and_pad


def apply_padding_refractive(
    image, data_dimensions: DataDimensions, padding_options: Padding, a0_log
):
    image = flip_and_pad(image, data_dimensions, padding_options, True)
    image.imag = image.imag - a0_log
    image = (
        image.real * data_dimensions.window + 1j * image.imag * data_dimensions.window
    )
    image.imag = image.imag + a0_log
    return image


def apply_l1(values, weight):
    values.real = torch.sign(values.real) * torch.maximum(
        torch.abs(values.real) - weight.real,
        torch.tensor(0, dtype=values.real.dtype, device=values.real.device),
    )
    values.imag = torch.sign(values.imag) * torch.maximum(
        torch.abs(values.imag) - weight.imag,
        torch.tensor(0, dtype=values.real.dtype, device=values.real.device),
    )
    return values


def apply_domain_constraint(values, phase_min=None, phase_max=None, absorption_min=None, absorption_max=None):
    if phase_min is not None or phase_max is not None:  # Apply phase constraints
        values.real = torch.clamp(values.real, min=phase_min, max=phase_max)
    if absorption_min is not None or absorption_max is not None:  # Apply absorption constraints
        values.imag = torch.clamp(values.imag, min=absorption_min, max=absorption_max)
    return values


def apply_filter(values, filter_kernel_real, filter_kernel_imag):
    values_real_fft = torch.fft.rfft2(values.real)
    values_real_fft *= filter_kernel_real
    values.real = torch.fft.irfft2(values_real_fft, values.real.size())

    values_imag_fft = torch.fft.rfft2(values.imag)
    values_imag_fft *= filter_kernel_imag
    values.imag = torch.fft.irfft2(values_imag_fft, values.imag.size())

    return values


def apply_window(values, window, intensities_log):
    values.imag = values.imag - intensities_log
    values *= window
    values.imag = values.imag + intensities_log

    return values
