import random
from typing import List
import numpy as np
import time
import os
from tqdm import tqdm
import logging

import holowizard.core.src
from holowizard.core.src.logging.logger import Logger
from holowizard.core.src.parameters.dask_options import DaskOptions
from holowizard.core.src.reconstruction.contexts.context_projection_multistage import ContextProjectionMultistage
from holowizard.core.src.reconstruction.contexts.context_projection_singlestage import ContextProjectionSingleStage
from holowizard.core.src.reconstruction.contexts.context_projection_set import ContextProjectionsSet
from holowizard.core.src.utils.transform import crop_center
from holowizard.core.src.reconstruction.controller.io.controller_single_stage_context_io import ControllerSingleStageContextIO


class ControllerBunchIO:
    def __init__(self):
        self.controller_single_stage_context_io = ControllerSingleStageContextIO()
        return
    
    def apply_for_keys(self, context, functions, keys, *args, verbose=True, **kwargs):
        iteratable_keys = tqdm(keys) if verbose else keys
        for key in iteratable_keys:
            for i in range(len(context.projection_contexts)):
                if context.projection_contexts[i].key == key:
                    if context.projection_contexts[i].current_projection_singlestage is not None:
                        for function in functions:
                            function(context.projection_contexts[i].current_projection_singlestage, *args, **kwargs)
                    break

    def draw_random_keys(self, context: ContextProjectionsSet, num: int) -> List[str]:
        index_list = [i for i in range(len(context.projection_contexts))]
        random.shuffle(index_list)
        keys = []
        for i in range(num):
            keys.append(context.projection_contexts[index_list[i]].key)
        return keys

    def get_current_bunch_size(self, context: ContextProjectionsSet):
        return context.bunch_size * context.current_options.padding.down_sampling_factor

    def get_bunch_keys(self, context: ContextProjectionsSet, bunch_index: int):
        if bunch_index * self.get_current_bunch_size(context) >= len(context.projection_contexts):
            return None
        keys = []
        for j in range(self.get_current_bunch_size(context)):
            if bunch_index * self.get_current_bunch_size(context) + j >= len(context.projection_contexts):
                break
            keys.append(context.projection_contexts[bunch_index * self.get_current_bunch_size(context) + j].key)

        return keys

    def write_bunch(self, context: ContextProjectionsSet, keys,verbose=True):
        self.apply_for_keys(context=context,functions=[
                            self.controller_single_stage_context_io.swap_oref_predicted_from_ram_to_file,
                            self.controller_single_stage_context_io.swap_measurements_from_ram_to_file
                            ]
                            ,keys=keys,verbose=verbose)
        #self.apply_for_keys(context=context,function=self.controller_single_stage_context_io.swap_probe_refractive_from_ram_to_file,keys=keys,verbose=verbose)


    def read_bunch(self, context: ContextProjectionsSet, keys, verbose=True):
        self.apply_for_keys(context=context,functions=[
                            self.controller_single_stage_context_io.swap_measurements_from_file_to_ram,
                            self.controller_single_stage_context_io.swap_oref_predicted_from_file_to_ram
                            ],keys=keys,verbose=verbose)
        #self.apply_for_keys(context,self.controller_single_stage_context_io.swap_probe_refractive_from_file_to_ram,keys=keys,verbose=verbose)


    def get_num_bunches(self, context: ContextProjectionsSet):
        return int(np.ceil(len(context.reco_params.measurements) / self.get_current_bunch_size(context)))

    def clean_bunch_from_ram(self, context: ContextProjectionsSet, keys):
        self.apply_for_keys(context=context,functions=[self.controller_single_stage_context_io.clean_arrays],keys=keys, verbose=False)

    def clean_context_file(self, context: ContextProjectionSingleStage, dask_options: DaskOptions) -> None:
        file_name = "projection_context_" + context.key + ".txt"
        file_path = dask_options.working_dir / file_name
        try:
            os.remove(file_path)
        except Exception:
            pass

    def write_context_file(
        self, current_stage_context: ContextProjectionSingleStage, dask_options: DaskOptions
    ) -> None:
        file_name = "projection_context_" + current_stage_context.key + ".txt"
        file_path = dask_options.working_dir / file_name
        self.clean_context_file(current_stage_context, dask_options)
        with open(file_path, "w") as file:
            file.write(current_stage_context.to_json())

    def read_context_file(
        self, key: str, dask_options: DaskOptions, timeout=1000, device=holowizard.core.src.torch_running_device
    ) -> ContextProjectionSingleStage:
        file_name = "projection_context_" + key + ".txt"
        file_path = dask_options.working_dir / file_name

        tries = 0
        while not file_path.exists() and tries < timeout:
            time.sleep(0.1)
            tries += 1
        if tries == timeout:
            raise RuntimeError("Reached timeout while trying to deserialize " + str(file_path))
        try:
            with open(file_path, "r") as f:
                context_serialized = f.read()
            context = ContextProjectionSingleStage.from_json(context_serialized, device=device)
        except Exception:
            raise RuntimeError("Could not deserialize file with key " + str(key) + " and file name " + str(file_path))

        return context

    def log_results_bunched(self, context: ContextProjectionsSet, logging_prefix):
        if Logger.current_log_level <= Logger.level_num_image_info:
            logging.info("Log results")
            num_bunches = self.get_num_bunches(context)
            for i in range(num_bunches):
                keys = self.get_bunch_keys(context, i)
                if keys is None:
                    return
                logging.info("Bunch " + str(i + 1) + "/" + str(num_bunches))
                self.read_bunch(context, keys)

                for key in keys:
                    for projection_context in context.projection_contexts:
                        if projection_context.key == key:
                            result_phaseshift = (
                                (projection_context.current_projection_singlestage.oref_predicted.real).cpu().numpy()
                            )
                            result_absorption = (
                                (projection_context.current_projection_singlestage.oref_predicted.imag).cpu().numpy()
                            )

                            logging.image_info(
                                logging_prefix + "_phaseshift_" + key,
                                result_phaseshift,
                            )
                            logging.image_info(
                                logging_prefix + "_absorption_" + key,
                                result_absorption,
                            )

                            logging.image_info(
                                logging_prefix + "_phaseshift_cropped_" + key,
                                crop_center(
                                    result_phaseshift,
                                    projection_context.current_projection_singlestage.data_dimensions.fov_size,
                                ),
                            )
                            logging.image_info(
                                logging_prefix + "_absorption_cropped_" + key,
                                crop_center(
                                    result_absorption,
                                    projection_context.current_projection_singlestage.data_dimensions.fov_size,
                                ),
                            )

                            logging.image_info(
                                logging_prefix + "_intensities_cropped",
                                crop_center(
                                    np.exp(-1 * result_absorption),
                                    projection_context.current_projection_singlestage.data_dimensions.fov_size,
                                ),
                            )

                            break