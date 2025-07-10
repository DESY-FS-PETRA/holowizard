import torch

import holowizard.core


def get_var(x):
    return torch.var(x).item()


def get_spec(x):
    sample_grid = torch.meshgrid(
        torch.fft.fftfreq(x.shape[0], device=holowizard.core.torch_running_device),
        torch.fft.fftfreq(x.shape[1], device=holowizard.core.torch_running_device),
    )

    xi, eta = sample_grid

    # Butterworth bandpass
    denom1 = xi * xi + eta * eta - 0.35**2
    denom2 = 1 + (0.09 / denom1) ** 10
    bandpass = 1 - 1 / denom2

    # x_fft_band = torch.fft.fft2(x) * bandpass
    x_fft_band = torch.fft.fft2(x)
    x[0, 0] = 0

    ones = torch.ones(x_fft_band.shape, device=x.device)

    spec = torch.log(ones + torch.abs(x_fft_band)).sum()

    return spec.item()


def get_gra(x):
    grad_x = x - torch.roll(x, 1, 0)
    grad_y = x - torch.roll(x, 1, 1)

    gc = torch.sqrt(grad_x**2 + grad_y**2).sum()

    return gc.item()


def get_lap(x):
    sum_x = torch.roll(x, 1, 0) + torch.roll(x, -1, 0)
    sum_y = torch.roll(x, 1, 1) + torch.roll(x, -1, 1)

    lp = ((sum_x + sum_y - 4 * x) ** 2).sum()

    return lp.item()


def get_gog(x):
    diff_0 = torch.abs(x[1:, :] - x[:-1, :])
    diff_1 = torch.abs(x[:, 1:] - x[:, :-1])

    diff = torch.sqrt(
        diff_0[:, 0 : diff_0.shape[1] - 1] ** 2
        + diff_1[0 : diff_1.shape[0] - 1, :] ** 2
    )

    diff_sorted, diff_sorted_indices = torch.sort(torch.flatten(diff))

    diff_sum = torch.sum(diff_sorted)

    N = len(diff_sorted)

    indices = torch.arange(1, N + 1, device=diff_sum.device)

    temp = (N + 0.5 - indices) / N

    temp = diff_sorted / diff_sum * temp

    gog = 1 - 2 * temp.sum()

    return gog.item()


def get_tog(x):
    diff_0 = torch.abs(x[1:, :] - x[:-1, :])
    diff_1 = torch.abs(x[:, 1:] - x[:, :-1])

    diff = torch.sqrt(
        diff_0[:, 0 : diff_0.shape[1] - 1] ** 2
        + diff_1[0 : diff_1.shape[0] - 1, :] ** 2
    )

    tog = torch.sqrt(torch.var(diff) / torch.mean(diff))

    return tog.item()
