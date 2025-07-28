import cupy
import torch
import numpy as np
import logging
from typing import List
import cupy as cp
import cupyx.scipy.fft as cufft
import scipy.fft

import holowizard.core


class FresnelPropagator:
    def __init__(self, fresnel_numbers: List[float], data_shape, running_device):
        # https: // docs.nvidia.com / cuda / cufft / index.html  # cufft-callback-routines
        self.fft_callback = r"""
                #include <cupy/complex.cuh>


extern "C" __global__ void FresnelCallback(
                                complex<float> *data, //this is thrust::complex<float>
                                int numel,
                                int height,
                                int width, 
                                float Fr
                                ){

    int offset = blockDim.x*blockIdx.x+threadIdx.x;

    //Prevents kernel to calculate something outside the image vector.
    if(offset>=numel){
    return;
    }

    complex<float> element = data[offset];
    // compute kernel on the fly

    size_t x_ = offset % width;
    size_t y_ = offset / width;
    long x = (x_ + (width/2))%width  - (width/2);
    long y = (y_ + (height/2))%height  - (height/2);

    float kx2 = 1.*x / float(width);
    kx2 *= kx2;
    float ky2 = 1.*y / float(height);
    ky2 *= ky2;
    float v = (-M_PI / (Fr))*(kx2 + ky2);
    float cosv;
    float sinv;
    sincosf(v, &sinv, &cosv);

    thrust::complex<float> kernelEl(cosv/numel, sinv/numel);

    //printf("i am working: x=%d, y=%d, x_=%d, y_=%d, val=(%f,%f)\n",
    // x ,y, x_, y_, cosv, sinv);

    data[offset] = element * kernelEl;
}   
                """

        self.num_distances = len(fresnel_numbers)
        self.fresnel_numbers = fresnel_numbers
        self.fresnel_kernel = cp.RawKernel(self.fft_callback, "FresnelCallback", translate_cucomplex=True)

    def propagate_forward(self, x, distance):
        # with torch.cuda.nvtx.range("forward propagator"):
        x = cp.fft.fft2(cp.from_dlpack(x))

        numel = np.prod(x.shape)
        threads_per_block = 1024
        blocks = cp.uint32((numel + threads_per_block - 1) / threads_per_block)

        self.fresnel_kernel(
            (blocks,),
            (threads_per_block,),
            (
                x,
                cp.int32(numel),
                cp.int32(x.shape[0]),
                cp.int32(x.shape[1]),
                cp.float32(self.fresnel_numbers[distance]),
            ),
        )

        x = cp.fft.ifft2(x, norm="forward")
        # print('after fft type:' + str(x.dtype))
        x = torch.from_dlpack(x)

        return x

    def propagate_forward_all(self, x):
        return [self.propagate_forward(x, distance) for distance in range(self.num_distances)]

    def propagate_back(self, x, distance):
        # with torch.cuda.nvtx.range("backward propagator"):
        # torch.cuda.nvtx.range_push("backward propagator")
        # print('kernel type:' + str(self.fresnel_kernels[distance].dtype))
        # print('before fft type:' + str(x.dtype))

        x = cp.fft.fft2(cp.from_dlpack(x))

        numel = x.shape[0] * x.shape[1]
        threads_per_block = 1024
        blocks = cp.uint32((numel + threads_per_block - 1) / threads_per_block)

        self.fresnel_kernel(
            (blocks,),
            (threads_per_block,),
            (
                x,
                cp.int32(numel),
                cp.int32(x.shape[0]),
                cp.int32(x.shape[1]),
                cp.float32(-self.fresnel_numbers[distance]),
            ),
        )

        x = cp.fft.ifft2(x, norm="forward")
        # print('after fft type:' + str(x.dtype))
        x = torch.from_dlpack(x)

        return x

    def propagate_back_all(self, x, distance):
        propagated = [self.propagate_back(x, distance) for distance in range(self.num_distances)]
        return propagated

    def get_measurements(self, x, distance):
        return torch.abs(self.propagate_forward(x, distance))

    def get_measurements_from_propagated_all(self, x):
        return [torch.abs(x[distance]) for distance in range(self.num_distances)]

    def get_measurements_all(self, x):
        return [self.get_measurements(x, distance) for distance in range(self.num_distances)]
