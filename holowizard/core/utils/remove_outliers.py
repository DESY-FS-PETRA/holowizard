import math
import torch
import numpy as np

from .median_pool_2d import MedianPool2d


def remove_outliers(input_image, threshold=1, filter_size=5):
    if type(input_image) == np.ndarray:
        # convert to torch tensor
        input_image = torch.tensor(input_image)
    image = input_image.clone()

    median_pool = MedianPool2d(kernel_size=filter_size, padding=int(math.floor(filter_size / 2)))

    filtered_image = median_pool.forward(image[None, None, :, :])[0, 0, :, :]
    diff_image = image - filtered_image
    std_dev_value = torch.std(diff_image)

    pixels_to_correct = torch.where(abs(diff_image) > (threshold * std_dev_value))
    image[pixels_to_correct] = filtered_image[pixels_to_correct]

    return image


def remove_outliers_multiprocess_wrapper(input_image):
    input_image_torch = torch.tensor(input_image.astype(np.float32), device="cuda")
    result = remove_outliers(input_image_torch)
    return result.cpu().numpy()
