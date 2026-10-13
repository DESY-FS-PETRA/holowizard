import logging
import torch

from holowizard.core.parameters.beam_setup import BeamSetup
from holowizard.core.parameters.data_dimensions import DataDimensions
from holowizard.core.parameters.padding import Padding
from holowizard.core.utils.transform import resize_arrays_to_size

from .flip_and_pad import flip_and_pad_and_fade

def process_beam_setup(beam_setup: BeamSetup, padding_options: Padding, data_dimensions: DataDimensions):
    beam_setup.px_size = beam_setup.px_size * padding_options.down_sampling_factor

    if beam_setup.flat_field is not None:
        mean = torch.mean(beam_setup.flat_field)
        beam_setup.flat_field = flip_and_pad_and_fade(beam_setup.flat_field, padding_options, data_dimensions, padval=mean)

    if beam_setup.probe_refractive is not None:
        print(beam_setup.probe_refractive.shape)
        mean_real = torch.mean(beam_setup.probe_refractive.real)
        mean_imag = torch.mean(beam_setup.probe_refractive.imag)
        probe_real = flip_and_pad_and_fade(
            beam_setup.probe_refractive.real, padding_options, data_dimensions, padval=mean_real,index=1
        )
        probe_imag = flip_and_pad_and_fade(
            beam_setup.probe_refractive.imag, padding_options, data_dimensions, padval=mean_imag,index=2
        )
        logging.image_info("probe_image_flipped_pad_fade",probe_imag.cpu().numpy())

        logging.image_info("probe_before_resize",beam_setup.probe_refractive.real.cpu().numpy())

        beam_setup.probe_refractive = probe_real + 1j * probe_imag
        logging.image_info("probe_after_resize", beam_setup.probe_refractive.real.cpu().numpy())
        print(beam_setup.probe_refractive.shape)


    return beam_setup

def process_beam_setup_noflipandpad_cropped(beam_setup: BeamSetup, padding_options: Padding, data_dimensions: DataDimensions):
    beam_setup.px_size = beam_setup.px_size * padding_options.down_sampling_factor

    resized_arrays = resize_arrays_to_size(data_dimensions.fov_size, 
                                           flat_field = beam_setup.flat_field,
                                           probe_refractive = beam_setup.probe_refractive)

    beam_setup.flat_field = resized_arrays["flat_field"]
    beam_setup.probe_refractive = resized_arrays["probe_refractive"]

    return beam_setup


def process_beam_setup_noflipandpad(beam_setup: BeamSetup, padding_options: Padding, data_dimensions: DataDimensions):
    beam_setup.px_size = beam_setup.px_size * padding_options.down_sampling_factor

    resized_arrays = resize_arrays_to_size(data_dimensions.total_size, 
                                           flat_field = beam_setup.flat_field,
                                           probe_refractive = beam_setup.probe_refractive)

    beam_setup.flat_field = resized_arrays["flat_field"]
    beam_setup.probe_refractive = resized_arrays["probe_refractive"]

    return beam_setup
