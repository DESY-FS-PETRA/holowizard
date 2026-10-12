import torch
import logging

import holowizard.core.src
from holowizard.core.src.methods.reconstruction.constraints import regularization
from holowizard.core.src.methods.reconstruction.gradients.analytical import get_gradient
from holowizard.core.src.methods.reconstruction.contexts.context_projection_singlestage import ContextProjectionSingleStage
from holowizard.core.src.methods.reconstruction.utils.filter import get_filter_kernels
from holowizard.core.src.utils.transform import crop_center


if "cuda" in holowizard.core.src.torch_running_device_name:
    from holowizard.core.src.models.fresnel_propagator_cupy import FresnelPropagatorCupy as FresnelPropagatorImpl
else:
    from holowizard.core.src.models.fresnel_propagator_torch import FresnelPropagatorTorch as FresnelPropagatorImpl


def print_infos(iteration, context: ContextProjectionSingleStage, force_update):
    if context.viewer is None:
        return

    if force_update or (iteration) % context.options.verbose_interval == 0:
        oref_predicted = context.oref_predicted
        probe_refractive = context.beam_setup.probe_refractive
        loss_cpu = context.se_losses_all
        for view in context.viewer:
            view.update(
                iteration,
                context.measurements[0].data,
                None,
                oref_predicted,
                probe_refractive,
                context.data_dimensions,
                loss_cpu,
            )


def reconstruct_single_stage(context: ContextProjectionSingleStage):
    with torch.no_grad():
        model = FresnelPropagatorImpl(
            context.fresnel_numbers, context.data_dimensions.total_size, holowizard.core.src.torch_running_device_name
        )

        (
            filter_kernel_obj_phase,
            filter_kernel_obj_absorption,
        ) = get_filter_kernels(
            context.options.regularization_object.gaussian_filter_fwhm,
            context.data_dimensions.total_size,
            holowizard.core.src.torch_running_device_name,
        )

        filter_kernel_vt, _ = get_filter_kernels(
            context.options.nesterov_object.gaussian_filter_fwhm,
            context.data_dimensions.total_size,
            holowizard.core.src.torch_running_device_name,
        )

        iteration = None
        for iteration in range(context.options.regularization_object.iterations):
            context.oref_predicted = regularization.apply_padding_refractive(
                context.oref_predicted,
                context.data_dimensions,
                context.options.padding,
                context.absorption_min,
            )

            context.nesterov_vt = regularization.apply_filter(
                context.nesterov_vt,
                filter_kernel_vt,
                filter_kernel_vt,
            )
            context.oref_predicted = regularization.apply_filter(
                context.oref_predicted,
                filter_kernel_obj_phase,
                filter_kernel_obj_absorption,
            )

            oref_predicted_old = context.oref_predicted.detach()
            torch.add(
                input=context.oref_predicted,
                other=context.nesterov_vt,
                alpha=-context.options.nesterov_object.update_rate,
                out=context.oref_predicted,
            )

            grad, loss = get_gradient(
                model=model,
                measurements=context.measurements,
                data_dimensions=context.data_dimensions,
                oref_predicted=context.oref_predicted,
                probe=torch.exp(1j * context.beam_setup.probe_refractive),
            )

            context.nesterov_vt = (
                context.options.nesterov_object.update_rate * context.nesterov_vt
                + context.options.regularization_object.update_rate * grad.real + 1j* context.options.regularization_object.update_rate * grad.imag
            )

            if context.options.regularization_object.l2_weight.imag != 0.0:
                context.nesterov_vt.imag = context.nesterov_vt.imag + (
                    context.options.regularization_object.l2_weight.imag
                    * context.oref_predicted.imag
                    / (context.oref_predicted.imag.norm(p=2) + 0.000001)
                )

            context.oref_predicted = oref_predicted_old - context.nesterov_vt
            context.oref_predicted = regularization.apply_domain_constraint(
                context.oref_predicted,
                phase_min=context.phaseshift_min,
                phase_max=context.phaseshift_max,
                absorption_min=context.absorption_min,
                absorption_max=context.absorption_max,
            )

            context.se_losses_all[context.current_iter_offset + iteration] = loss
            print_infos(context.current_iter_offset + iteration, context, force_update=False)

        if iteration is not None:
            print_infos(context.current_iter_offset + iteration, context, force_update=True)
