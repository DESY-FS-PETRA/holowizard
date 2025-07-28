from holowizard.core.reconstruction.constraints.regularization import apply_filter
import torch
import pytest

TINY = (5, 5)
SMALL = (128, 128)
PRODUCTION = (2048, 2048)


def old_filter(values, filter_kernel_real, filter_kernel_imag):
    """
    Apply a filter to complex valued field.

    Parameters
    ----------
    values : array_like
    The values to apply the filter to.
    filter_kernel_real : array_like
    The real part of the filter kernel.
    filter_kernel_imag : array_like
    The imaginary part of the filter kernel.

    Returns
    -------
    array_like
    The filtered values.

    Examples
    --------
    >>> values = torch.randn(5, 5, dtype=torch.complex64)
    >>> filter_kernel_real = torch.randn(5, 3)
    >>> filter_kernel_imag = torch.randn(5, 3)
    >>> filtered_values = apply_filter(values, filter_kernel_real, filter_kernel_imag)
    >>> filtered_values.shape
    torch.Size([5, 5])
    """

    values_real_fft = torch.fft.rfft2(values.real)
    values_real_fft *= filter_kernel_real
    values.real = torch.fft.irfft2(values_real_fft, values.real.size())

    values_imag_fft = torch.fft.rfft2(values.imag)
    values_imag_fft *= filter_kernel_imag
    values.imag = torch.fft.irfft2(values_imag_fft, values.imag.size())

    return values


# def joined_apply_filter(values, filter_kernel):
#     """
#     Apply a filter to complex valued field.

#     Parameters
#     ----------
#     values : array_like
#     The values to apply the filter to.
#     filter_kernel_real : array_like
#     The real part of the filter kernel.
#     filter_kernel_imag : array_like
#     The imaginary part of the filter kernel.

#     Returns
#     -------
#     array_like
#     The filtered values.

#     Examples
#     --------
#     >>> values = torch.randn(5, 5, dtype=torch.cfloat)
#     >>> filter_kernel = torch.randn(5, 3, dtype=torch.cfloat)
#     >>> filtered_values = joined_apply_filter(values, filter)
#     >>> filtered_values.shape
#     torch.Size([5, 5])
#     """

#     values_fft = torch.fft.fft2(values)
#     values_fft *= filter_kernel
#     values = torch.fft.ifft2(values_fft, values.size())

#     return values


def test_tiny_apply_filter():
    exp = torch.ones(TINY, dtype=torch.cfloat)

    assert exp.shape == TINY
    assert exp.real.shape == TINY
    assert exp.imag.shape == TINY

    filter_ = torch.zeros([5, 3], dtype=torch.cfloat)

    obs = apply_filter(exp, filter_.real, filter_.imag)

    assert obs.sum() == 0


def test_tiny_apply_filter_cross():
    exp = torch.complex(torch.ones(TINY), torch.ones(TINY) * 2)

    assert exp.shape == TINY
    assert exp.real.shape == TINY
    assert exp.imag.shape == TINY

    fshape = (TINY[0], (TINY[-1] // 2) + 1)
    filter_ = torch.complex(torch.ones(fshape), torch.zeros(fshape))

    obs1 = apply_filter(exp, filter_, filter_)
    obs2 = old_filter(
        exp,
        filter_.real[:, : (TINY[-1] // 2) + 1],
        filter_.imag[:, : (TINY[-1] // 2) + 1],
    )

    assert torch.allclose(obs1, obs2)


def test_small_apply_filter():
    exp = torch.ones(SMALL, dtype=torch.cfloat)

    assert exp.shape == SMALL

    fshape = (SMALL[0], (SMALL[-1] // 2) + 1)
    filter_ = torch.zeros(fshape, dtype=torch.cfloat)
    obs = apply_filter(exp, filter_, filter_)

    assert obs.sum() == 0


def test_small_old_apply_filter():
    exp = torch.ones(SMALL, dtype=torch.cfloat)

    assert exp.shape == SMALL

    fshape = (SMALL[0], (SMALL[-1] // 2) + 1)
    filter_ = torch.zeros(fshape, dtype=torch.cfloat)
    obs = old_filter(exp, filter_.real, filter_.imag)

    assert obs.sum() == 0


# BENCHMARK UNIT TESTS
# run by calling `python -m pytest . -m 'gpu' --durations=0`
# Note: will require access to cuda GPUs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPUs detected")
@pytest.mark.gpu
@pytest.mark.parametrize("shape", [SMALL, PRODUCTION])
def test_mass_apply_filter(shape):
    exp = torch.ones(shape, dtype=torch.cfloat)

    assert exp.shape == shape
    assert exp.real.shape == shape
    assert exp.imag.shape == shape

    fshape = (shape[0], (shape[-1] // 2) + 1)
    filter_ = torch.zeros(fshape, dtype=torch.cfloat)

    exp = exp.to(torch.device("cuda:0"))
    filter_ = filter_.to(torch.device("cuda:0"))

    obs = apply_filter(exp, filter_, filter_)

    assert obs.sum() == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPUs detected")
@pytest.mark.gpu
@pytest.mark.parametrize("shape", [SMALL, PRODUCTION])
def test_mass_old_filter(shape):
    exp = torch.ones(shape, dtype=torch.cfloat)

    assert exp.shape == shape
    assert exp.real.shape == shape
    assert exp.imag.shape == shape

    filter_dims = list(shape)
    filter_dims[-1] = (filter_dims[-1] // 2) + 1
    filter_ = torch.zeros(filter_dims, dtype=torch.cfloat)

    assert filter_.shape == torch.Size(filter_dims)

    exp = exp.to(torch.device("cuda:0"))
    filter_ = filter_.to(torch.device("cuda:0"))

    obs = old_filter(exp, filter_.real, filter_.imag)

    assert obs.sum() == 0.0
