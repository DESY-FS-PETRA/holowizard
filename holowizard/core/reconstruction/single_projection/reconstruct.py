import torch

from holowizard.core.reconstruction.constraints import regularization
from holowizard.core.reconstruction.gradients.analytical import get_gradient
from holowizard.core.reconstruction.single_projection.context import Context


def print_infos(iteration, context: Context, force_update):
    if context.viewer is None:
        return

    if force_update or (iteration) % context.current_options.verbose_interval == 0:
        oref_predicted = context.oref_predicted
        probe_refractive = context.beam_setup.probe_refractive
        loss_cpu = context.se_losses_all
        for view in context.viewer:
            view.update(
                iteration,
                oref_predicted,
                probe_refractive,
                context.data_dimensions,
                loss_cpu,
            )


def reconstruct(context: Context):
    with torch.no_grad():
        iteration = None
        for iteration in range(
            context.current_options.regularization_object.iterations
        ):

            context.oref_predicted = regularization.apply_padding_refractive(
                context.oref_predicted,
                context.data_dimensions,
                context.current_options.padding,
                context.absorption_min,
            )

            context.nesterov_vt = regularization.apply_filter(
                context.nesterov_vt,
                context.filter_kernel_vt,
                context.filter_kernel_vt,
            )
            context.oref_predicted = regularization.apply_filter(
                context.oref_predicted,
                context.filter_kernel_obj_phase,
                context.filter_kernel_obj_absorption,
            )

            oref_predicted_old = context.oref_predicted.detach()
            torch.add(
                input=context.oref_predicted,
                other=context.nesterov_vt,
                alpha=-context.current_options.nesterov_object.update_rate,
                out=context.oref_predicted,
            )

            grad, loss = get_gradient(
                model=context.model,
                measurements=context.measurements,
                data_dimensions=context.data_dimensions,
                oref_predicted=context.oref_predicted,
                probe=torch.exp(1j * context.beam_setup.probe_refractive),
            )

            context.nesterov_vt = (
                context.current_options.nesterov_object.update_rate
                * context.nesterov_vt
                + context.current_options.regularization_object.update_rate * grad
            )

            if context.current_options.regularization_object.l2_weight.imag != 0.0:
                context.nesterov_vt.imag = context.nesterov_vt.imag + (
                    context.current_options.regularization_object.l2_weight.imag
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
            print_infos(
                context.current_iter_offset + iteration, context, force_update=False
            )

        if iteration is not None:
            print_infos(context.current_iter_offset + iteration, context, force_update=True)
