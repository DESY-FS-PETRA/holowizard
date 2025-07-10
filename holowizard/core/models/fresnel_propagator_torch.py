import torch
import numpy as np
import logging
from typing import List
import cupy as cp
import cupyx.scipy.fft as cufft
import scipy.fft

import holowizard.core
from holowizard.interface.parameters.measurement import Measurement


class FresnelPropagator:
    def __init__(self, fresnel_numbers: List[float], data_shape, running_device):
        self.num_distances = len(fresnel_numbers)
        self.fresnel_numbers = fresnel_numbers

        sample_grid = torch.meshgrid(
            torch.fft.fftfreq(data_shape[0], device=running_device),
            torch.fft.fftfreq(data_shape[1], device=running_device),
            indexing="ij",
        )
        xi, eta = sample_grid

        kernel_func = lambda distance: torch.exp(
            (-1j * np.pi) / self.fresnel_numbers[distance] * (xi * xi + eta * eta)
        )
        kernel_func_conj = lambda distance: torch.exp(
            (-1j * np.pi) / (-self.fresnel_numbers[distance]) * (xi * xi + eta * eta)
        )

        self.fresnel_kernels = [
            kernel_func(distance).to(running_device)
            for distance in range(self.num_distances)
        ]
        self.fresnel_kernels_conj = [
            kernel_func_conj(distance).to(running_device)
            for distance in range(self.num_distances)
        ]

        logging.info("Initializing Fresnel propagation model with fresnel numbers:")
        [
            logging.info("Fr: " + str(self.fresnel_numbers[distance]))
            for distance in range(self.num_distances)
        ]

    def propagate_forward(self, x, distance):
        propagated = torch.fft.ifft2(torch.fft.fft2(x) * self.fresnel_kernels[distance])
        return propagated

    def propagate_forward_all(self, x):
        return [
            self.propagate_forward(x, distance)
            for distance in range(self.num_distances)
        ]

    def propagate_back(self, x, distance):
        propagated = torch.fft.ifft2(
            torch.fft.fft2(x) * self.fresnel_kernels_conj[distance]
        )
        return propagated

    def propagate_back_all(self, x, distance):
        propagated = [
            self.propagate_back(x, distance) for distance in range(self.num_distances)
        ]
        return propagated

    def get_measurements(self, x, distance):
        return torch.abs(self.propagate_forward(x, distance))

    def get_measurements_from_propagated_all(self, x):
        return [torch.abs(x[distance]) for distance in range(self.num_distances)]

    def get_measurements_from_propagated(self, x):
        return torch.abs(x)

    def get_measurements_all(self, x):
        return [
            self.get_measurements(x, distance) for distance in range(self.num_distances)
        ]
