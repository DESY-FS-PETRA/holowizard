import logging
import torchvision
import traceback
import socket
import cupy as cp

import torch
import gc
import time

from pathlib import Path

import holowizard.core.src
from holowizard.core.src.logging.logger import Logger
from holowizard.core.src.parameters.dask_options import DaskOptions
from holowizard.core.src.methods.preprocessing.flip_and_pad import flip_and_pad, flip_and_pad_and_fade
from holowizard.core.src.methods.reconstruction.modules.contexts.context_projection_singlestage import (
    ContextProjectionSingleStage,
)
from holowizard.core.src.methods.reconstruction.methods.asrm.reconstruct_single_stage import reconstruct_single_stage
from holowizard.core.src.api.viewer.loss_viewer import LossViewer
from holowizard.core.src.methods.reconstruction.utils.logging import log_params
from holowizard.core.src.misc.transform import crop_center
from holowizard.core.src.methods.preprocessing.process_data_dimensions import get_window_from_padding_options
from holowizard.core.src.methods.reconstruction.controller.io.controller_single_stage_context_io import (
    ControllerSingleStageContextIO,
)


def init_logger(dask_options, context):
    Logger.current_log_level = Logger.level_num_image_info
    Logger.configure(dask_options.working_dir / "logs/", "asrm_" + context.key)
    logging.info("task_run_asrm for key " + context.key)


def init_nesterov_vt(context: ContextProjectionSingleStage):
    context.nesterov_vt = torch.zeros(
        context.data_dimensions.total_size, device=holowizard.core.torch_running_device, dtype=torch.complex64
    )


def cleanup_nesterov_vt(context):
    context.nesterov_vt = None


def init_window(context):
    context.data_dimensions.window = get_window_from_padding_options(context.data_dimensions, context.options.padding)


def cleanup_window(context):
    context.data_dimensions.window = None


def flip_and_pad_measurements(context):
    for i in range(len(context.measurements)):
        padval = context.options.padding.prototype_field
        padval = 1.0 if padval is None else padval

        context.measurements[i].data = flip_and_pad_and_fade(
            context.measurements[i].data, context.options.padding, context.data_dimensions, padval=padval
        )


def crop_measurements(context):
    for i in range(len(context.measurements)):
        context.measurements[i].data = crop_center(context.measurements[i].data, context.data_dimensions.fov_size)


def cleanup_and_log_times(t, labels):
    gc.collect()
    torch.cuda.empty_cache()
    t.append(time.perf_counter())
    labels.append("stuff")

    timings = "\n".join(f"{labels[i]}: {t[i + 1] - t[i]:.3f}s" for i in range(len(labels)))
    logging.info("\n" + timings)


def log_images(context):
    if context.key == "0.0":
        logging.image_info("measurement", context.measurements[0].data.cpu().numpy())

        logging.image_info("probe_real", context.beam_setup.probe_refractive.real.cpu().numpy())
        logging.image_info("probe_imag", context.beam_setup.probe_refractive.imag.cpu().numpy())

        logging.image_info("probe_amplitudes", torch.exp(-context.beam_setup.probe_refractive.imag).cpu().numpy())
        logging.image_info(
            "probe_amplitudes_cropped",
            crop_center(torch.exp(-context.beam_setup.probe_refractive.imag), context.data_dimensions.fov_size)
            .cpu()
            .numpy(),
        )
        # logging.image_info("flat_field", torch.abs(context.beam_setup.flat_field).cpu().numpy())
        # logging.image_info("nesterov_real", context.nesterov_vt.real.cpu().numpy())


def handle_exception(exception):
    hostname = socket.gethostname()
    tb = traceback.format_exc()
    raise RuntimeError(f"Exception from {hostname}: {tb}") from exception


def free_gpu_memory():
    """Free GPU memory before running workload."""

    # 1. Free CuPy memory pools (what you already have)
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    # 2. Also free any per-device memory pools (important in multi-GPU setups)
    for device_id in range(cp.cuda.runtime.getDeviceCount()):
        with cp.cuda.Device(device_id):
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    # 3. If PyTorch is in your environment, clear its cache too
    try:
        import torch

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except ImportError:
        pass

    # 5. Synchronize to ensure all async CUDA ops are done
    cp.cuda.Device().synchronize()

    # 6. Python GC last — frees Python objects that hold GPU references
    gc.collect()


def task_run_asrm_lazyflipandpad(dask_options_serialized: str, context_single_stage_serialized: str):
    free_gpu_memory()
    try:
        io_controller = ControllerSingleStageContextIO()
        t, labels = [], []
        t.append(time.perf_counter())

        dask_options: DaskOptions = DaskOptions.from_json(dask_options_serialized)
        t.append(time.perf_counter())
        labels.append("get dask options")
        context = ContextProjectionSingleStage.from_json(context_single_stage_serialized)
        t.append(time.perf_counter())
        labels.append("get context")

        init_logger(dask_options, context)

        io_controller.swap_arrays_from_file_to_ram(context)
        t.append(time.perf_counter())
        labels.append("load arrays from file")

        init_window(context)
        t.append(time.perf_counter())
        labels.append("get window")

        init_nesterov_vt(context)
        t.append(time.perf_counter())
        labels.append("create nesterov_vt")

        log_images(context)
        t.append(time.perf_counter())
        labels.append("log stuff")

        flip_and_pad_measurements(context)
        t.append(time.perf_counter())
        labels.append("flip pad measurement")

        log_params(context.measurements, context.beam_setup, [context.options], context.data_dimensions)
        t.append(time.perf_counter())
        labels.append("log params")

        context.viewer = [LossViewer()]

        reconstruct_single_stage(context)
        t.append(time.perf_counter())
        labels.append("reconstruct single stage")

        context.oref_predicted = crop_center(context.oref_predicted, context.data_dimensions.fov_size)
        t.append(time.perf_counter())
        labels.append("crop center oref")

        io_controller.swap_oref_predicted_from_ram_to_file(context)
        t.append(time.perf_counter())
        labels.append("write oref to file")

        cleanup_nesterov_vt(context)
        t.append(time.perf_counter())
        labels.append("cleanup nesterov_vt")

        cleanup_window(context)
        t.append(time.perf_counter())
        labels.append("cleanup window")

        io_controller.clean_arrays(context)
        t.append(time.perf_counter())
        labels.append("cleanup context")

        cleanup_and_log_times(t, labels)

        return context.to_json()
    except Exception as e:
        handle_exception(e)
