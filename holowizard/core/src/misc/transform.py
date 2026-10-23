import torch
import numpy as np
from torch.nn.functional import pad
import torchvision.transforms as ttf
from torchvision.transforms import InterpolationMode


def crop_center(img, crop: tuple):
    cropx = crop[0]
    cropy = crop[1]

    x, y = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    if startx < 0:
        startx = 0
        cropx = img.shape[0]
    if starty < 0:
        starty = 0
        cropy = img.shape[1]
    return img[startx : (startx + cropx), starty : (starty + cropy)]


def pad_to_size(img, output_size, padval, pad_mode="constant"):
    padsize = (np.floor((np.array(output_size) - np.array(img.shape)) / 2)).astype(int)
    padsize2 = (np.array(output_size) - (np.array(img.shape) + 2 * padsize)).astype(int)

    result = pad(img, (padsize[1], padsize[1], padsize[0], padsize[0]), pad_mode, value=padval)

    if any(padsize2 > 0):
        result = pad(result, (padsize2[1], 0, padsize2[0], 0), mode=pad_mode, value=padval)

    return result


def resize_arrays_to_size(new_size, **kwargs):
    resizer = ttf.Resize(new_size, interpolation=InterpolationMode.BILINEAR, antialias=True)

    resized_arrays = {}
    for key, value in kwargs.items():
        if value is not None:
            if torch.is_complex(value):
                resized_arrays[key] = (
                    resizer(value.real[None, None, :, :])[0, 0, :, :]
                    + 1j * resizer(value.imag[None, None, :, :])[0, 0, :, :]
                )
            else:
                resized_arrays[key] = resizer(value[None, None, :, :])[0, 0, :, :]
        else:
            resized_arrays[key] = None

    return resized_arrays
