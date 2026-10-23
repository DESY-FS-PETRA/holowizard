import torch
import logging
from holowizard.core.src.parameters.data_dimensions import DataDimensions
from holowizard.core.src.parameters.padding import Padding
from holowizard.core.src.methods.preprocessing.flip_and_pad import resize_and_flip_and_pad_and_fade
from holowizard.core.src.misc.transform import resize_arrays_to_size


def process_measurement(image, padding_options: Padding, data_dimensions: DataDimensions, index):
    image = torch.rot90(image / padding_options.a0)
    return resize_and_flip_and_pad_and_fade(image, padding_options, data_dimensions, index=index, padval=1.0)


def process_measurement_noflipandpad(image, padding_options: Padding, data_dimensions: DataDimensions, index):
    image = torch.rot90(image / padding_options.a0)

    resized_arrays = resize_arrays_to_size(data_dimensions.fov_size, image=image)

    image = resized_arrays["image"]

    logging.image_debug("image_resized_" + str(index), image.cpu().numpy())

    return image
