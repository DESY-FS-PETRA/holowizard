from typing import List
import logging
import numpy as np

from holowizard.core.logging.logger import Logger
from holowizard.core.utils.transform import crop_center
from holowizard.core.reconstruction.contexts.context_projection_multistage import ContextProjectionMultistage


def log_preprocessed_params(beam_setup, data_dimensions):
    logging.info(f"{'BeamSetup':<17}{beam_setup.to_log_json()}")
    logging.info(f"{'DataDimensions':<17}{data_dimensions.to_log_json()}")


def log_input(measurements):
    i = 0
    for measurement in measurements:
        if Logger.current_log_level <= Logger.level_num_image_info:
            logging.image_info("reconstruct_x_input_" + str(i), measurement.data.cpu().numpy())
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


def log_result(logging_prefix, result_phaseshift, result_absorption, fov_size, index):
    logging.image_info(logging_prefix + "_phaseshift_" + str(index).zfill(4), result_phaseshift)
    logging.image_info(logging_prefix + "_absorption_" + str(index).zfill(4), result_absorption)
    logging.image_info(logging_prefix + "_intensities", np.exp(-1 * result_absorption))

    '''
    logging.image_info(
        logging_prefix + "_phaseshift_cropped_" + str(index).zfill(4),
        crop_center(result_phaseshift, fov_size),
    )
    logging.image_info(
        logging_prefix + "_absorption_cropped_" + str(index).zfill(4),
        crop_center(result_absorption, fov_size),
    )
    '''
    
    logging.image_info(logging_prefix + "_intensities_cropped", crop_center(np.exp(-1 * result_absorption), fov_size))


def log_results(logging_prefix, context_list: List[ContextProjectionMultistage]):
    if Logger.current_log_level <= Logger.level_num_image_info:
        for i in range(len(context_list)):
            result_phaseshift = (context_list[i].current_projection_singlestage.oref_predicted.real).cpu().numpy()
            result_absorption = (context_list[i].current_projection_singlestage.oref_predicted.imag).cpu().numpy()

            log_result(
                logging_prefix,
                result_phaseshift,
                result_absorption,
                context_list[i].current_projection_singlestage.data_dimensions.fov_size,
                i,
            )
        # logging.image_info(logging_prefix + "_intensities_cropped", crop_center(torch.exp(-1 * result_absorption),data_dimensions.fov_size))
