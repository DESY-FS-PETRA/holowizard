import logging

from holowizard.core.logging.logger import Logger
from holowizard.core.utils.transform import *


def log_preprocessed_params(beam_setup, data_dimensions):
    logging.info(f"{'BeamSetup':<17}{beam_setup.to_log_json()}")
    logging.info(f"{'DataDimensions':<17}{data_dimensions.to_log_json()}")


def log_input(measurements):
    i = 0
    for measurement in measurements:
        if Logger.current_log_level <= Logger.level_num_image_info:
            logging.image_info(
                "reconstruct_x_input_" + str(i), measurement.data.cpu().numpy()
            )
        i = i + 1


def log_params(measurements, beam_setup, options, data_dimensions):
    i = 0
    for measurement in measurements:
        logging.info(f"{'Measurement[' + str(i) + ']':<17}{measurement.to_log_json()}")
        i = i + 1

    i = 0
    for option in options:
        logging.info(f"{'Option[' + str(i) + '] ':<17}{option.to_log_json()}")
        i = i + 1

    log_preprocessed_params(beam_setup, data_dimensions)


def log_results(logging_prefix, data, data_dimensions):
    if Logger.current_log_level <= Logger.level_num_image_info:
        for i in range(len(data)):

            result_phaseshift = (data[i].real).cpu().numpy()
            result_absorption = (data[i].imag).cpu().numpy()

            logging.image_debug(
                logging_prefix + "_phaseshift_" + str(i).zfill(4), result_phaseshift
            )
            logging.image_debug(
                logging_prefix + "_absorption_" + str(i).zfill(4), result_absorption
            )
            # logging.image_debug(logging_prefix + "_intensities", torch.exp(-1 *
            #                                                           result_absorption))

            logging.image_info(
                logging_prefix + "_phaseshift_cropped_" + str(i).zfill(4),
                crop_center(result_phaseshift, data_dimensions.fov_size),
            )
            logging.image_info(
                logging_prefix + "_absorption_cropped_" + str(i).zfill(4),
                crop_center(result_absorption, data_dimensions.fov_size),
            )
        # logging.image_info(logging_prefix + "_intensities_cropped", crop_center(torch.exp(-1 * result_absorption),data_dimensions.fov_size))
