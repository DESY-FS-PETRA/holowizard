import torch
from holowizard.core.serialization.params_serializer import ParamsSerializer
from holowizard.core.reconstruction.probe_retrieval.worker.worker_object import (
    reconstruct,
)
from holowizard.core.reconstruction.probe_retrieval.worker.worker_probe import (
    get_probe_gradient,
)
from holowizard.core.reconstruction.probe_retrieval.host_context import HostContext


def print_infos(host_context: HostContext):
    if host_context.viewer is None:
        return

    oref_predicted_cpu = ParamsSerializer.deserialize(
        host_context.dask_options.working_dir + "oref_predicted_" + str(0) + ".pkl"
    ).cpu()

    for view in host_context.viewer:
        view.update(
            host_context.current_iter_offset,
            oref_predicted_cpu,
            host_context.beam_setup.probe_refractive,
            host_context.data_dimensions,
            host_context.se_losses_all,
        )


def basic_update(host_context: HostContext, function, update_oref=True):
    future_results = []

    for j in range(len(host_context.measurements)):
        args_dict = {
            "working_dir": host_context.dask_options.working_dir,
            "index": j,
            "options": host_context.current_options.to_json(),
            "iter_offset": host_context.current_iter_offset,
            "update_oref": update_oref,
        }

        future_results.append(
            host_context.dask_controller.client.submit(function, args_dict)
        )

    return future_results


def object_update(host_context: HostContext):
    future_results = basic_update(
        host_context=host_context, function=reconstruct, update_oref=True
    )
    se_losses, _ = host_context.read_intermediate_results(future_results)
    host_context.se_losses_all = torch.cat(
        (host_context.se_losses_all, se_losses.cpu())
    )
    host_context.current_iter_offset = (
        host_context.current_iter_offset
        + host_context.current_options.regularization_object.iterations
    )


def probe_init(host_context: HostContext):
    future_results = basic_update(
        host_context=host_context, function=reconstruct, update_oref=False
    )
    host_context.beam_setup.probe_refractive, se_losses = probe_gradient_step(
        host_context, future_results
    )
    host_context.current_iter_offset = host_context.current_iter_offset + 1

    if host_context.se_losses_all is None:
        host_context.se_losses_all = se_losses[-1].unsqueeze(0)
    else:
        host_context.se_losses_all = torch.cat(
            (host_context.se_losses_all, se_losses[-1].unsqueeze(0))
        )

    HostContext.write_inputs(
        dask_options=host_context.dask_options, beam_setup=host_context.beam_setup
    )


def probe_gradient_step(host_context: HostContext, results):
    se_losses, grad_probe = host_context.read_intermediate_results(results)

    probe_refractive = host_context.beam_setup.probe_refractive.to(torch.device("cpu"))

    probe_refractive.real = (
        probe_refractive.real
        - host_context.current_options.regularization_probe.update_rate
        * grad_probe.real
    )
    probe_refractive.imag = (
        probe_refractive.imag
        - host_context.current_options.regularization_probe.update_rate
        * grad_probe.imag
    )

    probe_refractive.imag = torch.minimum(
        probe_refractive.imag, 0 * torch.tensor(0, device=host_context.torch_device)
    )
    # probe_refractive = apply_filter(
    #    probe_refractive, host_context.filter_kernel_probe_phase, host_context.filter_kernel_probe_absorption
    # )

    return probe_refractive, se_losses


def probe_update(host_context: HostContext):
    for i in range(host_context.current_options.regularization_probe.iterations):
        future_results = basic_update(
            host_context, get_probe_gradient, update_oref=False
        )
        host_context.beam_setup.probe_refractive, se_losses = probe_gradient_step(
            host_context, future_results
        )

        HostContext.write_inputs(
            dask_options=host_context.dask_options, beam_setup=host_context.beam_setup
        )

        host_context.current_iter_offset = host_context.current_iter_offset + 1
        host_context.se_losses_all = torch.cat(
            (host_context.se_losses_all, se_losses.cpu())
        )

        print_infos(host_context)
