import numpy as np
import torch
from holowizard.core.utils.remove_outliers import remove_outliers
import holowizard.core

import matplotlib
import matplotlib.pyplot as plt


def components_model_to_tensors(components_model):
    components_model.mean_ = torch.tensor(
        components_model.mean_, device=holowizard.core.torch_running_device
    )
    components_model.components_ = torch.tensor(
        components_model.components_, device=holowizard.core.torch_running_device
    )

    return components_model


def get_synthetic_flatfield(image, components_model):
    orig_shape = image.shape

    components_model = components_model_to_tensors(components_model)

    synthetic_flat_field = image.reshape(np.prod(image.shape)) - components_model.mean_
    synthetic_flat_field = torch.matmul(
        components_model.components_.float(), synthetic_flat_field.float()
    )
    synthetic_flat_field = torch.matmul(
        torch.transpose(components_model.components_.float(), 0, 1),
        synthetic_flat_field.float(),
    )
    synthetic_flat_field += components_model.mean_
    return synthetic_flat_field.reshape(orig_shape)


def correct_flatfield_old(image, components_model):
    image = image.clone()
    components_model = components_model_to_tensors(components_model)

    synthetic_flat_field = image.reshape(np.prod(image.shape)) - components_model.mean_
    synthetic_flat_field = torch.matmul(
        components_model.components_, synthetic_flat_field
    )
    synthetic_flat_field = torch.matmul(
        torch.transpose(components_model.components_, 0, 1), synthetic_flat_field
    )
    synthetic_flat_field += components_model.mean_
    corrected_image = image / torch.reshape(synthetic_flat_field, image.shape)

    corrected_image[torch.where(corrected_image < 10 * torch.finfo(float).eps)] = (
        10 * torch.finfo(float).eps
    )

    corrected_image = remove_outliers(corrected_image)

    return corrected_image


def correct_flatfield(image, components):
    image = image.clone()
    components = components_model_to_tensors(components)

    image[torch.where(image < 1e-7)] = 1e-7
    cur_data = torch.log(image.reshape(np.prod(image.shape)))

    synthetic_flat_field = get_synthetic_flatfield(cur_data, components)

    corrected_image = torch.reshape(cur_data - synthetic_flat_field, image.shape)

    corrected_image[torch.where(corrected_image < -20)] = -20

    corrected_image = remove_outliers(torch.exp(corrected_image))

    return corrected_image
