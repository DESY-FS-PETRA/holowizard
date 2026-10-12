import torch
import logging
from typing import List
from copy import deepcopy
from tqdm import tqdm
import os
from pathlib import Path


import holowizard.core.src
from holowizard.core.src.parameters.dask_options import DaskOptions
from holowizard.core.src.parameters.reco_params import RecoParams
from holowizard.core.src.parameters.measurement import Measurement
from holowizard.core.src.reconstruction.viewer.viewer import Viewer
from holowizard.core.src.reconstruction.contexts.context_probe_multistage import ContextProbeMultistage
from holowizard.core.src.preprocessing.process_padding_options import process_beam_setup_noflipandpad, process_data_dimensions
from holowizard.core.src.preprocessing.process_data_dimensions import get_window_from_padding_options
from holowizard.core.src.reconstruction.controller.dask_utils.slurm_cluster import SlurmCluster
from holowizard.core.src.reconstruction.contexts.context_projection_set import ContextProjectionsSet
from holowizard.core.src.reconstruction.controller.io.controller_bunch_io import ControllerBunchIO
from holowizard.core.src.reconstruction.controller.io.controller_beam_setup_io import ControllerBeamSetupIO
from holowizard.core.src.reconstruction.controller.controller_projection_set import ControllerProjectionSet
from holowizard.core.src.reconstruction.constraints import regularization
from holowizard.core.src.reconstruction.utils.filter import get_filter_kernels
from holowizard.core.src.utils.transform import resize_arrays_to_size



from holowizard.core.src.models.fresnel_propagator_torch import FresnelPropagatorTorch as FresnelPropagatorImpl
from holowizard.core.src.reconstruction.gradients.analytical import get_gradient

from holowizard.core.src.preprocessing.flip_and_pad import flip_and_pad_and_fade
from holowizard.core.src.utils.transform import crop_center

class ControllerProbeMultistage:
    def __init__(self, working_dir:Path):
        self.working_dir = working_dir
        self.controller_beam_setup_io = ControllerBeamSetupIO()
        self.controller_bunch_io = ControllerBunchIO()
        self.controller_projection_set = ControllerProjectionSet(working_dir=working_dir)

    def create_context_probe_multistage(
        self,
        dask_options: DaskOptions,
        reco_params: RecoParams,
        viewer: List[Viewer] = None,
        cluster: SlurmCluster = None,
    ):
        logging.info("Creating probe retrieval context")
        context_probe_retrieval = ContextProbeMultistage(
            key="",
            viewer=viewer,
            reco_params=reco_params,
            current_stage=-1,
            bunch_size=min(50, len(reco_params.measurements)),
            current_iter_offset=0,
            dask_options=dask_options,
            torch_device=holowizard.core.src.torch_running_device,
            cluster=cluster,
            se_losses_all=torch.empty(0, device=holowizard.core.src.torch_running_device),
            current_global_beam_setup=deepcopy(reco_params.beam_setup),
            current_options=None,
            current_data_dimensions=None,
        )

        context_probe_retrieval.current_global_beam_setup.probe_path = self.controller_beam_setup_io.get_default_probe_path(self.working_dir)

        logging.info("Contexts creation finished")
        return context_probe_retrieval

    def set_stage(self, context: ContextProbeMultistage, stage_index):
        if context.current_stage == stage_index:
            return

        context.current_stage = stage_index
        context.current_options = context.reco_params.reco_options[stage_index]

        logging.info("Set stage of probe context")

        context.current_data_dimensions = process_data_dimensions(
            deepcopy(context.reco_params.data_dimensions),
            deepcopy(context.reco_params.reco_options[stage_index].padding),
        )

        resized_arrays = resize_arrays_to_size(context.current_data_dimensions.total_size, 
                                           probe_refractive = context.current_global_beam_setup.probe_refractive)

        
        context.current_global_beam_setup = process_beam_setup_noflipandpad(
            deepcopy(context.reco_params.beam_setup),
            deepcopy(context.current_options.padding),
            context.current_data_dimensions,
        )

        context.current_global_beam_setup.probe_path = self.controller_beam_setup_io.get_default_probe_path(self.working_dir)
        context.current_global_beam_setup.probe_refractive = resized_arrays["probe_refractive"]


        try:
            logging.image_info("probe_stage_" + str(stage_index), context.current_global_beam_setup.probe_refractive.imag.cpu().numpy())
        except:
            pass

        if context.current_global_beam_setup.probe is None:
            self.init_empty_probe(context)

        self.controller_beam_setup_io.write_probe_refractive(context.current_global_beam_setup)

        #print("probe memory")
        #memory_breakdown(context.beam_setup)

    def make_probe_gradient_step(
        self, context_probe: ContextProbeMultistage, context_projection_set: ContextProjectionsSet, keys: List[str]
    ):
        model = None

        logging.info("Calculating task locally")
        se_losses = []
        pbar = tqdm(total=len(keys))
        grad = torch.zeros_like(context_probe.current_global_beam_setup.probe_refractive)
        ones = torch.ones_like(context_probe.current_global_beam_setup.probe_refractive)
        ff_measurement = [Measurement(z01=0,angle=0,data_path="",data=context_probe.current_global_beam_setup.flat_field)]
        for key in keys:
            for projection_context in context_projection_set.projection_contexts:
                if projection_context.key == key:

                    if model is None:
                        model = FresnelPropagatorImpl(
                            projection_context.current_projection_singlestage.fresnel_numbers, projection_context.current_projection_singlestage.data_dimensions.total_size,
                            context_probe.torch_device,
                            verbose=False
                        )


                    grad_key, loss = get_gradient(
                        model=model,
                        measurements=projection_context.current_projection_singlestage.measurements,
                        data_dimensions=projection_context.current_projection_singlestage.data_dimensions,
                        oref_predicted=context_probe.current_global_beam_setup.probe_refractive,
                        probe=torch.exp(1j * projection_context.current_projection_singlestage.oref_predicted),
                    )

                    grad_ff, _ = get_gradient(
                        model=model,
                        measurements=ff_measurement,
                        data_dimensions=projection_context.current_projection_singlestage.data_dimensions,
                        oref_predicted=context_probe.current_global_beam_setup.probe_refractive,
                        probe=ones,
                    )

                    grad += grad_key
                    grad += 2*grad_ff


                    se_losses.append(loss)
                    pbar.update(1)

        ''' For zyla
        context_probe.current_beam_setup.probe_refractive = context_probe.current_beam_setup.probe_refractive -(
                (context_probe.current_options.regularization_probe.update_rate
                * grad.real + 
                0.001*1j*context_probe.current_options.regularization_probe.update_rate
                * grad.imag)

                / len(keys)
        )
        '''

        # For diamond lenses
        '''
        context_probe.current_beam_setup.probe_refractive = context_probe.current_beam_setup.probe_refractive -(
                (context_probe.current_options.regularization_probe.update_rate
                * grad.real + 
                1j*context_probe.current_options.regularization_probe.update_rate
                * grad.imag)

                / len(keys)
        )
        '''
        context_probe.current_global_beam_setup.probe_refractive = context_probe.current_global_beam_setup.probe_refractive -(
                context_probe.current_options.regularization_probe.update_rate
                * grad)/ len(keys)
        
        
        (
            filter_kernel_probe_phase,
            filter_kernel_probe_absorption,
        ) = get_filter_kernels(
            context_projection_set.current_options.regularization_probe.gaussian_filter_fwhm,
            context_projection_set.current_data_dimensions.total_size,
            holowizard.core.src.torch_running_device,
        )

        context_probe.current_global_beam_setup.probe_refractive = regularization.apply_filter(
                    context_probe.current_global_beam_setup.probe_refractive,
                    filter_kernel_probe_phase,
                    filter_kernel_probe_absorption,
                    )
        
        #context_probe.current_beam_setup.probe_refractive = torch.clamp(context_probe.current_beam_setup.probe_refractive.real,min=0) + 1j* context_probe.current_beam_setup.probe_refractive.imag

        propagated_probe_amplitudes = model.get_measurements(context_probe.current_global_beam_setup.probe,0)

        return sum(se_losses) / len(keys), propagated_probe_amplitudes

    def run_update_probe_for_keys(
        self, context_probe: ContextProbeMultistage, context_projection_set: ContextProjectionsSet, keys: List[str]
    ):
        
        (
            filter_kernel_obj_phase,
            filter_kernel_obj_absorption,
        ) = get_filter_kernels(
            context_projection_set.current_options.regularization_object.gaussian_filter_fwhm,
            context_projection_set.current_data_dimensions.total_size,
            holowizard.core.src.torch_running_device_name,
        )

        object_to_plot = None
        measurement_to_plot = None
        window = None
        for key in keys:
            for projection_context in context_projection_set.projection_contexts:
                if projection_context.key == key:
                    if window is None:
                        window = get_window_from_padding_options(projection_context.current_projection_singlestage.data_dimensions, projection_context.current_projection_singlestage.options.padding)
                    
                    #print(projection_context.current_projection_singlestage.data_dimensions.to_json())
                    #print(projection_context.current_projection_singlestage.oref_predicted.shape)
                    projection_context.current_projection_singlestage.data_dimensions.window = window
                    #print("--------------------------------------------------------------------------------------------------")
                    #print(projection_context.current_projection_singlestage.oref_predicted.shape)
                    projection_context.current_projection_singlestage.oref_predicted = regularization.apply_padding_refractive(
                    projection_context.current_projection_singlestage.oref_predicted,
                    projection_context.current_projection_singlestage.data_dimensions,
                    projection_context.current_projection_singlestage.options.padding,
                    projection_context.current_projection_singlestage.absorption_min,
                    )
                    #print(projection_context.current_projection_singlestage.oref_predicted.shape)
                        
                    '''
                    projection_context.current_projection_singlestage.oref_predicted = regularization.apply_filter(
                    projection_context.current_projection_singlestage.oref_predicted,
                    filter_kernel_obj_phase,
                    filter_kernel_obj_absorption,
                    )
                    '''
                    
                    for i in range(len(projection_context.current_projection_singlestage.measurements)):
                        projection_context.current_projection_singlestage.measurements[i].data = flip_and_pad_and_fade(
                            projection_context.current_projection_singlestage.measurements[i].data,
                            projection_context.current_projection_singlestage.options.padding,
                            projection_context.current_projection_singlestage.data_dimensions,
                            padval=1.0
                        )

                    if object_to_plot is None:
                        object_to_plot = projection_context.current_projection_singlestage.oref_predicted

                    if measurement_to_plot is None:
                        measurement_to_plot = projection_context.current_projection_singlestage.measurements[0].data

        for i in range(context_probe.current_options.regularization_probe.iterations):
            logging.info(
                "Probe update iteration "
                + str(i + 1)
                + "/"
                + str(context_probe.current_options.regularization_probe.iterations)
            )
            se_losses, propagated_probe_amplitudes = self.make_probe_gradient_step(context_probe, context_projection_set, keys)
            context_probe.se_losses_all = torch.cat((context_probe.se_losses_all, se_losses.reshape(1)))
            context_probe.current_iter_offset = len(context_probe.se_losses_all) - 1
    
        
            if i%10 ==0:
                for view in context_probe.viewer:
                    view.update(
                        context_probe.current_iter_offset,
                        measurement_to_plot,
                        propagated_probe_amplitudes,
                        object_to_plot,
                        context_probe.current_global_beam_setup.probe_refractive,
                        context_probe.current_data_dimensions,
                        context_probe.se_losses_all,
                    )
            

        for key in keys:
            for projection_context in context_projection_set.projection_contexts:
                if projection_context.key == key:
                    projection_context.current_projection_singlestage.oref_predicted = crop_center(
                    projection_context.current_projection_singlestage.oref_predicted,
                    context_projection_set.current_data_dimensions.fov_size)
                    for i in range(len(projection_context.current_projection_singlestage.measurements)):
                        projection_context.current_projection_singlestage.measurements[i].data = crop_center(
                        projection_context.current_projection_singlestage.measurements[i].data,
                        projection_context.current_projection_singlestage.data_dimensions.fov_size
                        )


    def run_update_probe(self, context_probe: ContextProbeMultistage, context_projection_set: ContextProjectionsSet):
        for i in range(context_projection_set.current_options.prototype_field):
            keys = self.controller_bunch_io.draw_random_keys(context_projection_set, context_probe.bunch_size)
            logging.info("Reading bunch from files")
            self.controller_bunch_io.read_bunch(context_projection_set, keys)
            logging.info("Update probe with bunch")
            self.run_update_probe_for_keys(context_probe, context_projection_set, keys)
            self.controller_bunch_io.clean_bunch_from_ram(context_projection_set, keys)
            logging.info("Write probe")
            self.controller_beam_setup_io.write_probe_refractive(context_probe.current_global_beam_setup)
            self.controller_bunch_io.clean_bunch_from_ram(context_projection_set, keys)

    def init_probe(self, context_probe: ContextProbeMultistage, context_projection_set: ContextProjectionsSet):
        logging.info("Draw random keys")
        keys = self.controller_bunch_io.draw_random_keys(context_projection_set, context_probe.bunch_size)
        logging.info("Read bunch")
        self.controller_bunch_io.read_bunch(context_projection_set, keys)
        print(context_projection_set.projection_contexts[0].current_projection_singlestage.oref_predicted.shape)
        logging.info("Run probe update loop")
        self.run_update_probe_for_keys(context_probe, context_projection_set, keys)
        self.controller_beam_setup_io.write_probe_refractive(context_probe.current_global_beam_setup)
        self.controller_bunch_io.clean_bunch_from_ram(context_projection_set, keys)

        logging.info("Run object update")
        self.controller_projection_set.prepare_next_reco_block_for_keys(context_projection_set,keys)
        context_projection_set.cluster.start_cluster()
        self.controller_projection_set.run_object_update_for_keys(context_projection_set, keys)
        context_projection_set.cluster.stop_cluster()
        logging.info("Run probe update loop")
        self.controller_bunch_io.read_bunch(context_projection_set, keys)
        self.run_update_probe_for_keys(context_probe, context_projection_set, keys)
        self.controller_beam_setup_io.write_probe_refractive(context_probe.current_global_beam_setup)
        self.controller_projection_set.reset_projection_contexts(context_projection_set)
        self.controller_bunch_io.clean_bunch_from_ram(context_projection_set, keys)

    def init_empty_probe(self, context: ContextProbeMultistage):
        context.current_global_beam_setup.probe_refractive = torch.zeros(context.current_data_dimensions.total_size)