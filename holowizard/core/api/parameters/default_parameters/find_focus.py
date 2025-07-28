from copy import deepcopy

from holowizard.core.api.parameters import Padding, Options, Regularization


def get_default_options(a0=0.98, phase_max=0.0, padding_options=None):
    if padding_options is None:
        padding_options = Padding(
            padding_mode=Padding.PaddingMode.MIRROR_ALL,
            padding_factor=4.0,
            down_sampling_factor=16,
            cutting_band=0,
            a0=a0,
        )

    options_warmup = Options(
        regularization_object=Regularization(
            iterations=700,
            update_rate=0.9,
            l2_weight=0.0 + 0.1 * 1j,
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
            l2_weight=0.0 + 0.1 * 1j,
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
            l2_weight=0.0 + 0.01 * 1j,
            gaussian_filter_fwhm=2.0 + 8.0j,
        ),
        nesterov_object=Regularization(update_rate=1.0, gaussian_filter_fwhm=16.0 + 16.0j),
        verbose_interval=100,
        padding=deepcopy(padding_options),
    )

    options_upscale_4.padding.down_sampling_factor = 4
    options_upscale_4_lowreg.padding.down_sampling_factor = 4

    return [options_warmup, options_upscale_4, options_upscale_4_lowreg]
