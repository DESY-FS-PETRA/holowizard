import torch
import socket

from holowizard.core.reconstruction.gradients.analytical import get_gradient
from holowizard.core.reconstruction.probe_retrieval.worker.worker_context import (
    WorkerContext,
)


def get_probe_gradient(args_dict):
    torch.cuda.empty_cache()

    try:
        print("calculating one iteration")
        worker_context: WorkerContext = WorkerContext.from_dict(
            args_dict=args_dict, update_mode=WorkerContext.UpdateMode.PROBE
        )

        with torch.no_grad():
            worker_context.grad_probe, loss = get_gradient(
                model=worker_context.model,
                measurements=worker_context.measurements,
                data_dimensions=worker_context.data_dimensions,
                oref_predicted=worker_context.probe_refractive,
                probe=torch.exp(1j * worker_context.oref_predicted),
            )

        worker_context.se_loss_records[0] = loss
        worker_context.write_results()

        # Return index of processed projection
        result = {"index": worker_context.index}
    except RuntimeError as expt:
        torch.cuda.empty_cache()
        hostname = socket.gethostname()
        raise RuntimeError("Exception from " + str(hostname)) from expt

    return result
