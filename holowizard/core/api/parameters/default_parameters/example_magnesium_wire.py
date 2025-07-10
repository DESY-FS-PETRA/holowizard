from copy import deepcopy


def get_default_options(a0=0.98, phase_max=0.0, padding_options=None):

    if padding_options is None:
        padding_options = Padding(
            padding_mode=Padding.PaddingMode.MIRROR_ALL,
            padding_factor=4,
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
        nesterov_object=Regularization(
            update_rate=1.0, gaussian_filter_fwhm=4.0 + 4.0j
        ),
        verbose_interval=100,
        padding=deepcopy(padding_options),
    )

    options_upscale_4 = Options(
        regularization_object=Regularization(
            iterations=300,
            update_rate=1.1,
            l2_weight=0.0 + 0.01 * 1j,
            gaussian_filter_fwhm=2.0 + 8.0j,
        ),
        nesterov_object=Regularization(
            update_rate=1.0, gaussian_filter_fwhm=8.0 + 8.0j
        ),
        verbose_interval=100,
        padding=deepcopy(padding_options),
    )

    options_upscale_2 = Options(
        regularization_object=Regularization(
            iterations=500,
            update_rate=1.1,
            l2_weight=0.0 + 0.001 * 1j,
            gaussian_filter_fwhm=2.0 + 8.0j,
        ),
        nesterov_object=Regularization(
            update_rate=1.0, gaussian_filter_fwhm=32.0 + 32.0j
        ),
        verbose_interval=100,
        padding=deepcopy(padding_options),
        prototype_field=0.0,
    )

    options_mainrun = Options(
        regularization_object=Regularization(
            iterations=1500,
            update_rate=1.1,
            l2_weight=0.0 + 0.0 * 1j,
            gaussian_filter_fwhm=2.0 + 8.0j,
        ),
        nesterov_object=Regularization(update_rate=0.5, gaussian_filter_fwhm=complex(0)),
        verbose_interval=100,
        padding=deepcopy(padding_options),
        prototype_field=0.0,
    )

    options_upscale_4.padding.down_sampling_factor = 4
    options_upscale_2.padding.down_sampling_factor = 2
    options_mainrun.padding.down_sampling_factor = 2

    return [options_warmup, options_upscale_4, options_upscale_2, options_mainrun]
