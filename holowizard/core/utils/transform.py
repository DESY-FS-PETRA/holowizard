import numpy as np
from torch.nn.functional import pad


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


def pad_to_size(img, outputSize, padMode="constant", padval=1):
    padsize = (np.floor((np.array(outputSize) - np.array(img.shape)) / 2)).astype(int)
    padsize2 = (np.array(outputSize) - (np.array(img.shape) + 2 * padsize)).astype(int)

    result = pad(img, (padsize[1], padsize[1], padsize[0], padsize[0]), padMode, value=padval)

    if any(padsize2 > 0):
        result = pad(result, (padsize2[1], 0, padsize2[0], 0), mode=padMode, value=padval)

    return result
