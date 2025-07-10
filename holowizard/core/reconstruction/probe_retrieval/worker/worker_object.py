import torch
import socket

from holowizard.core.reconstruction.constraints import regularization
from holowizard.core.reconstruction.gradients.analytical import get_gradient
from holowizard.core.reconstruction.probe_retrieval.worker.worker_context import (
    WorkerContext,
)


def reconstruct(args_dict):
    torch.cuda.empty_cache()

    try:
        worker_context: WorkerContext = WorkerContext.from_dict(
            args_dict=args_dict, update_mode=WorkerContext.UpdateMode.OBJECT
        )
        print(worker_context.absorption_min)

        with torch.no_grad():
            for iteration in range(
                worker_context.options.regularization_object.iterations
            ):
                print("Iteration ", iteration)

                worker_context.oref_predicted = regularization.apply_padding_refractive(
                    worker_context.oref_predicted,
                    worker_context.data_dimensions,
                    worker_context.options.padding,
                    worker_context.absorption_min,
                )

                if iteration == 0:
                    print(worker_context.oref_predicted[0, 0])

                worker_context.nesterov_vt = regularization.apply_filter(
                    worker_context.nesterov_vt,
                    worker_context.filter_kernel_vt,
                    worker_context.filter_kernel_vt,
                )
                worker_context.oref_predicted = regularization.apply_filter(
                    worker_context.oref_predicted,
                    worker_context.filter_kernel_obj_phase,
                    worker_context.filter_kernel_obj_absorption,
                )

                worker_context.oref_predicted_old = (
                    worker_context.oref_predicted.detach()
                )
                torch.add(
                    input=worker_context.oref_predicted,
                    other=worker_context.nesterov_vt,
                    alpha=-worker_context.options.nesterov_object.update_rate,
                    out=worker_context.oref_predicted,
                )

                grad, loss = get_gradient(
                    model=worker_context.model,
                    measurements=worker_context.measurements,
                    data_dimensions=worker_context.data_dimensions,
                    oref_predicted=worker_context.oref_predicted,
                    probe=torch.exp(1j * worker_context.probe_refractive),
                )

                worker_context.nesterov_vt = (
                    worker_context.options.nesterov_object.update_rate
                    * worker_context.nesterov_vt
                    + worker_context.options.regularization_object.update_rate * grad
                )

                if worker_context.options.regularization_object.l2_weight.real != 0.0:
                    worker_context.nesterov_vt.real = (
                        worker_context.nesterov_vt.real
                        + (
                            worker_context.options.regularization_object.l2_weight.real
                            * worker_context.oref_predicted.real
                            / (worker_context.oref_predicted.real.norm(p=2) + 0.000001)
                        )
                    )

                if worker_context.options.regularization_object.l2_weight.imag != 0.0:
                    worker_context.nesterov_vt.imag = (
                        worker_context.nesterov_vt.imag
                        + (
                            worker_context.options.regularization_object.l2_weight.imag
                            * worker_context.oref_predicted.imag
                            / (worker_context.oref_predicted.imag.norm(p=2) + 0.000001)
                        )
                    )

                worker_context.oref_predicted = (
                    worker_context.oref_predicted_old - worker_context.nesterov_vt
                )

                worker_context.oref_predicted = regularization.apply_domain_constraint(
                    worker_context.oref_predicted,
                    phase_min=worker_context.phaseshift_min,
                    phase_max=worker_context.phaseshift_max,
                    absorption_min=worker_context.absorption_min,
                    absorption_max=worker_context.absorption_max,
                )

                worker_context.se_loss_records[iteration] = loss
                torch.cuda.nvtx.range_pop()

            # worker_context.nesterov_vt = torch.zeros_like(worker_context.nesterov_vt)

            # As last step: calculate gradient of probe
            worker_context.grad_probe, loss = get_gradient(
                model=worker_context.model,
                measurements=worker_context.measurements,
                data_dimensions=worker_context.data_dimensions,
                oref_predicted=worker_context.probe_refractive,
                probe=torch.exp(1j * worker_context.oref_predicted),
            )

        worker_context.write_results()

        # Return index of processed projection
        result = {"index": worker_context.index}
    except Exception as expt:
        torch.cuda.empty_cache()
        hostname = socket.gethostname()
        raise RuntimeError("Exception from " + str(hostname)) from expt

    return result
