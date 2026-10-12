import torch
import logging
from typing import List
from copy import deepcopy
from tqdm import tqdm
from multiprocessing.dummy import Pool
from itertools import repeat
import os
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


import holowizard.core
from holowizard.core.parameters.dask_options import DaskOptions
from holowizard.core.parameters.reco_params import RecoParams
from holowizard.core.reconstruction.viewer.viewer import Viewer
from holowizard.core.preprocessing.process_padding_options import process_data_dimensions 

from holowizard.core.reconstruction.contexts.context_projection_set import ContextProjectionsSet
from holowizard.core.reconstruction.contexts.context_projection_multistage import ContextProjectionMultistage
from holowizard.core.reconstruction.contexts.context_projection_singlestage import ContextProjectionSingleStage
from holowizard.core.reconstruction.controller.controller_projection_multistage_lazy import ControllerProjectionMultistageLazy as ControllerProjectionMultistage
from holowizard.core.reconstruction.controller.io.controller_bunch_io import ControllerBunchIO
from holowizard.core.reconstruction.controller.io.controller_single_stage_context_io import ControllerSingleStageContextIO
from holowizard.core.reconstruction.controller.dask_utils.slurm_cluster import SlurmCluster
from holowizard.core.reconstruction.controller.tasks.broadcast_task import broadcast_task
from holowizard.core.reconstruction.controller.tasks.task_run_asrm_lazyflipandpad import task_run_asrm_lazyflipandpad as task_run_asrm

from holowizard.core.reconstruction.controller.visualization import slider_2_3d_matrices
#from memory_profiler import profile

class ControllerProjectionSet:
    def __init__(self,working_dir:Path):
        self.start_stage = -1
        self.working_dir = working_dir
        self.controller_projection_multistage = ControllerProjectionMultistage()
        self.controller_bunch_io = ControllerBunchIO()
        return

    def create_context_projection_set(
        self,
        dask_options: DaskOptions,
        reco_params: RecoParams,
        viewer: List[Viewer] = None,
        torch_device=None,
        cluster: SlurmCluster = None,
        start_stage = -1,
        min_bunch_size=6500
    ):
        self.start_stage = start_stage

        if torch_device is None:
            torch_device = holowizard.core.torch_running_device 


        logging.info("Creating Projections context")
        context_projections = ContextProjectionsSet(
            key="",
            projection_contexts=None,
            bunch_size=min(min_bunch_size, len(reco_params.measurements)),
            viewer=viewer,
            reco_params=reco_params,
            torch_device=torch_device,
            current_stage=start_stage,
            current_iter_offset=0,
            dask_options=dask_options,
            cluster=cluster,
            se_losses_all=torch.empty(0, device=torch_device),
            current_global_beam_setup=None,
            current_options=None,
            current_data_dimensions=None,
        )


        logging.info("Creating projection contexts")
        self.reset_projection_contexts(context_projections)

        logging.info("Contexts creation finished")
        return context_projections
    
    def get_default_stage_context(self, context: ContextProjectionsSet):
        self.controller_bunch_io.read_bunch(context, [context.projection_contexts[0].key])
        return context.projection_contexts[0].current_projection_singlestage


    def get_stage_context(self, context: ContextProjectionsSet, key):
        for i in range(len(context.projection_contexts)):
            if context.projection_contexts[i].key == key:
                return context.projection_contexts[i].current_projection_singlestage

    def set_stage_of_projection_context(self, projection_context:ContextProjectionMultistage, keys:List[str], stage_index:int, working_dir, pbar=None):
        controller = ControllerProjectionMultistage()
        controller_io = ControllerSingleStageContextIO()
        for key in keys:
            if projection_context.key == key:
                controller.set_stage(
                    projection_context, stage_index, verbose=False
                )
                controller_io.define_io_paths(projection_context.current_projection_singlestage, working_dir)
                if pbar is not None:
                    pbar.update(1)
                break

        return projection_context

    def set_stage_all(self, context: ContextProjectionsSet, stage_index):
        if context.current_stage == stage_index or stage_index < 0:
            return

        context.current_stage = stage_index

        context.current_options = context.reco_params.reco_options[stage_index]
        processed_data_dimensions = process_data_dimensions(
            deepcopy(context.reco_params.data_dimensions),
            deepcopy(context.reco_params.reco_options[stage_index].padding),
            init_window=False
        )

        context.current_data_dimensions = processed_data_dimensions

        num_bunches = self.controller_bunch_io.get_num_bunches(context)
        for i in range(num_bunches):
            keys = self.controller_bunch_io.get_bunch_keys(context, i)
            if keys is None:
                return
            logging.info("Bunch " + str(i + 1) + "/" + str(num_bunches))

            if stage_index > 0:
                self.controller_bunch_io.read_bunch(context, keys)

            logging.info("Set stages of projection contexts")

            pool = Pool()
            with tqdm(total=len(keys)) as pbar:
                context.projection_contexts = list(pool.starmap(self.set_stage_of_projection_context,
                                                         zip(context.projection_contexts,
                                                             repeat(keys),
                                                             repeat(stage_index),
                                                             repeat(self.working_dir),
                                                             repeat(pbar))))

            '''
            pbar = tqdm(total=len(keys))
            for projection_context in context.projection_contexts:
                self.set_stage_of_projection_context(projection_context,keys,stage_index,self.working_dir,pbar=pbar)

            '''

            logging.info("Write stages")
            self.controller_bunch_io.write_bunch(context, keys)

        single_stage_context = self.get_default_stage_context(context)
        context.current_data_dimensions = single_stage_context.data_dimensions

    def set_stage_selected(self, context: ContextProjectionsSet, stage_index, selected_keys):
        logging.info("Set stages of selected projection contexts")
        context.current_stage = stage_index

        context.current_options = context.reco_params.reco_options[stage_index]
        processed_data_dimensions = process_data_dimensions(
            deepcopy(context.reco_params.data_dimensions),
            deepcopy(context.reco_params.reco_options[stage_index].padding),
        )

        context.current_data_dimensions = processed_data_dimensions

        num_bunches = self.controller_bunch_io.get_num_bunches(context)
        for i in range(num_bunches):
            keys = self.controller_bunch_io.get_bunch_keys(context, i)
            if keys is None:
                keys_intersected = selected_keys
                selected_projection_contexts = [
                    pc for pc in context.projection_contexts 
                    if pc.key in keys_intersected
                ]
            else:
                logging.info("Bunch " + str(i + 1) + "/" + str(num_bunches))

                keys_intersected = [item for item in keys if item in selected_keys]

                
                # Skip this bunch if no keys match
                if not keys_intersected:
                    continue

                if stage_index > 0:
                    self.controller_bunch_io.read_bunch(context, keys_intersected)

                # Filter projection contexts: must be in BOTH keys (current bunch) AND keys_intersected
                selected_projection_contexts = [
                    pc for pc in context.projection_contexts 
                    if pc.key in keys_intersected and pc.key in keys
                ]

            
            pool = Pool()
            with tqdm(total=len(keys_intersected)) as pbar:
                updated_contexts = list(pool.starmap(self.set_stage_of_projection_context,
                                                zip(selected_projection_contexts,
                                                    repeat(keys_intersected),
                                                    repeat(stage_index),
                                                    repeat(self.working_dir),
                                                    repeat(pbar))))

            # Update the projection contexts back
            context_dict = {pc.key: pc for pc in updated_contexts}
            for idx, pc in enumerate(context.projection_contexts):
                if pc.key in context_dict:
                    context.projection_contexts[idx] = context_dict[pc.key]

            logging.info("Write stages")
            self.controller_bunch_io.write_bunch(context, keys_intersected)

        single_stage_context = self.get_stage_context(context, selected_keys[0])
        context.current_data_dimensions = single_stage_context.data_dimensions

        #print("projection memory")
        #memory_breakdown(context.projection_contexts[0])

    def store_bunch_in_ram(
        self, context_projection_set: ContextProjectionsSet, single_stage_contexts: List[ContextProjectionSingleStage]
    ):
        for single_stage_context in single_stage_contexts:
            key = single_stage_context.key

            for i in range(len(context_projection_set.projection_contexts)):
                if context_projection_set.projection_contexts[i].key == key:
                    context_projection_set.projection_contexts[i].current_projection_singlestage = single_stage_context
                    break

    def update_loss_array(
        self, context_projection_set: ContextProjectionsSet, single_stage_contexts: List[ContextProjectionSingleStage]
    ):
        se_losses = []
        for single_stage_context in single_stage_contexts:
            (
                se_losses.append(
                    single_stage_context.se_losses_all[-single_stage_context.options.regularization_object.iterations :]
                )
            )

        se_losses_summed = torch.tensor(
            sum(se_losses) / len(single_stage_contexts), device=context_projection_set.torch_device
        )
        context_projection_set.se_losses_all = torch.cat((context_projection_set.se_losses_all, se_losses_summed))
        context_projection_set.current_iter_offset = len(context_projection_set.se_losses_all) - 1

    def create_multi_stage_context(self,measurement, beam_setup, reco_options, data_dimensions, output_path, session_params, initial_guess, torch_device, progress_bar=None):
        controller = ControllerProjectionMultistage()

        measurements = [measurement]
        reco_params = RecoParams(
            beam_setup=beam_setup,
            measurements=measurements,
            reco_options=reco_options,
            data_dimensions=data_dimensions,
            output_path=output_path,
            session_params=session_params,
            initial_guess=initial_guess,
        )

        context = controller.create_multistage_context(reco_params,torch_device=torch_device)
        
        if progress_bar is not None:
            progress_bar.update(1)

        return context

    def reset_projection_contexts(self, context: ContextProjectionsSet):
        pool = Pool()

        #memory_breakdown(context.projection_contexts)


  
        multi_stage_contexts = list(pool.starmap(self.create_multi_stage_context,
                                        zip(context.reco_params.measurements,
                                            repeat(context.reco_params.beam_setup),
                                            repeat(context.reco_params.reco_options),
                                            repeat(context.reco_params.data_dimensions),
                                            repeat(context.reco_params.output_path),
                                            repeat(context.reco_params.session_params),
                                            repeat(context.reco_params.initial_guess),
                                            repeat(context.torch_device),
                                            repeat(None))))

        #memory_breakdown(multi_stage_contexts[0])

        '''
        for i in tqdm(range(len(multi_stage_contexts))):
            self.controller_projection_multistage.set_stage(
                multi_stage_contexts[i], context.current_stage, verbose=False
            )

            if multi_stage_contexts[i].current_projection_singlestage is not None:
                self.controller_single_stage_context_io.define_io_paths(multi_stage_contexts[i].current_projection_singlestage,context.dask_options.working_dir)
                self.controller_single_stage_context_io.swap_arrays_from_ram_to_file(multi_stage_contexts[i].current_projection_singlestage)
                #multi_stage_contexts[i].current_projection_singlestage = None
        '''

        context.projection_contexts = multi_stage_contexts
        target_stage = context.current_stage
        context.current_stage = self.start_stage

        self.set_stage_all(context,target_stage)

        #memory_breakdown(context.projection_contexts[0])
        #print(context.projection_contexts[0].current_projection_singlestage.nesterov_vt)
        #print(context.projection_contexts[0].current_projection_singlestage.beam_setup.probe)

    def get_current_objects(self, context: ContextProjectionsSet, indices):
        logging.info("Gathering current objects")

        #objects = []
        #for key in tqdm(keys):
        #    for i in range(len(context.projection_contexts)):
        #        if context.projection_contexts[i].key == key:
        #            #self.controller_bunch_io.read_bunch(context, [key],verbose=False)
        #            path = context.projection_contexts[i].current_projection_singlestage.oref_predicted_path
        #            objects.append(torch.load(path, map_location=holowizard.core.torch_running_device)) 
        #            #self.controller_bunch_io.clean_bunch_from_ram(context, [key])

        def load_slice(i):
            path = context.projection_contexts[i].current_projection_singlestage.oref_predicted_path
            return torch.load(path, map_location=holowizard.core.torch_running_device)

        with ThreadPoolExecutor() as executor:
            objects = list(tqdm(executor.map(load_slice, indices), total=len(indices)))
    
        #objects = []
        #for i in tqdm(indices):
        #    path = context.projection_contexts[i].current_projection_singlestage.oref_predicted_path
        #    objects.append(torch.load(path, map_location=holowizard.core.torch_running_device))
   
        return objects

    def run_object_update_for_keys(self, context: ContextProjectionsSet, keys):
        self.prepare_next_reco_block_for_keys(context, keys)
        results = broadcast_task(context.cluster, context.projection_contexts, task_run_asrm, keys, device=context.torch_device)
        self.store_bunch_in_ram(context, results)
        self.update_loss_array(context, results)

    def prepare_next_reco_block_for_keys(self, context: ContextProjectionsSet, keys):
        for key in keys:
            for projection_context in context.projection_contexts:
                if projection_context.key == key:
                    self.controller_projection_multistage.prepare_next_reco_block(projection_context)

    def run_object_update(self, context: ContextProjectionsSet, keys=None):
        if keys is None:
            logging.info("Running object update for " + str(len(context.projection_contexts)) + " projections")
        else:
            logging.info("Running object update for " + str(len(keys)) + " projections")
        num_bunches = self.controller_bunch_io.get_num_bunches(context)
        context.cluster.start_cluster()
        for i in range(num_bunches):
            bunch_keys = self.controller_bunch_io.get_bunch_keys(context, i)
            if bunch_keys == None:
                keys_to_process = keys
            else:
                if keys is not None:
                    keys_to_process = [item for item in bunch_keys if item in keys]  # intersect with the selected keys
                else:
                    keys_to_process = bunch_keys
                if keys_to_process is None:
                    return
            logging.info("Bunch " + str(i + 1) + "/" + str(num_bunches))
            #logging.info("Reading bunch from files")
            #self.controller_bunch_io.read_bunch(context, keys_to_process)
            self.run_object_update_for_keys(context=context, keys=keys_to_process)
            #logging.info("Writing bunch to files")
            #self.controller_bunch_io.write_bunch(context, keys_to_process)
        context.cluster.stop_cluster()