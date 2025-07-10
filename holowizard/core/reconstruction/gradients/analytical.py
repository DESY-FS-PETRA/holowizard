import torch
from typing import List

from holowizard.core.parameters.data_dimensions import DataDimensions
from holowizard.core.parameters.measurement import Measurement


def get_gradient(
    model,
    measurements: List[Measurement],
    data_dimensions: DataDimensions,
    oref_predicted,
    probe,
):
    oref_predicted = oref_predicted.type(torch.complex64)
    object = torch.exp(1j * oref_predicted)
    object_conj = torch.conj(object)
    probe_conj = torch.conj(probe)

    object_propagated = model.propagate_forward_all(object * probe)
    predicted_holograms = model.get_measurements_from_propagated_all(object_propagated)

    num_measurements = len(measurements)

    loss = 0
    all_gradients = torch.zeros_like(oref_predicted)
    for distance in range(num_measurements):
        measured_hologram = measurements[distance].data
        detector_plane_diff = object_propagated[distance] - object_propagated[
            distance
        ] * torch.sqrt(measured_hologram / predicted_holograms[distance])

        single_gradient = (
            -1j
            * object_conj
            * probe_conj
            * model.propagate_back(detector_plane_diff, distance)
        )

        all_gradients.add_(single_gradient)

        loss = (
            torch.abs(
                predicted_holograms[distance][data_dimensions.fov_range]
                - measured_hologram[data_dimensions.fov_range]
            )
            .pow(2)
            .sum()
        )

    all_gradients.real = torch.nan_to_num(all_gradients.real) / num_measurements
    all_gradients.imag = torch.nan_to_num(all_gradients.imag) / num_measurements

    N = (data_dimensions.fov_range_raw[0][1] - data_dimensions.fov_range_raw[0][0]) * (
        data_dimensions.fov_range_raw[1][1] - data_dimensions.fov_range_raw[1][0]
    )

    return all_gradients, torch.from_dlpack(loss) / N / num_measurements
