from typing import List

from holowizard.interface.parameters.dask_options import DaskOptions
from holowizard.core.dask_utils.dask_controller import DaskController

from holowizard.interface.parameters.options import Options
from holowizard.interface.parameters.measurement import Measurement
from holowizard.interface.parameters.beam_setup import BeamSetup
from holowizard.interface.parameters.data_dimensions import DataDimensions

from holowizard.core.reconstruction.viewer import Viewer

from holowizard.core.reconstruction.logging import *
from holowizard.core.reconstruction.probe_retrieval.basic_operations import *
from holowizard.core.reconstruction.probe_retrieval.host_context import HostContext


def reconstruct(
    measurements: List[Measurement],
    beam_setup: BeamSetup,
    options: List[Options],
    data_dimensions: DataDimensions,
    dask_options: DaskOptions,
    viewer: List[Viewer],
):
    host_context = HostContext(
        dask_controller=DaskController(dask_options),
        dask_options=dask_options,
        viewer=viewer,
        options=options,
        measurements=measurements,
        beam_setup=beam_setup,
        data_dimensions=data_dimensions,
    )

    probe_init(host_context)
    object_update(host_context)

    for options_index in range(len(options)):

        host_context.set_stage(options_index)

        logging.image_debug(
            "oref_rescaled_init_"
            + str(options_index)
            + "_after_"
            + str(host_context.current_options.padding.down_sampling_factor),
            host_context.oref_predicted[0].real.cpu().numpy(),
        )

        logging.info(
            "Downsampling factor "
            + str(host_context.current_options.padding.down_sampling_factor)
        )

        print_infos(host_context)

        for update_block in range(host_context.current_options.update_blocks):
            logging.comment(
                "Block "
                + str(update_block + 1)
                + " / "
                + str(host_context.current_options.update_blocks)
            )
            probe_update(host_context)
            print_infos(host_context)
            object_update(host_context)
            print_infos(host_context)

        probe_update(host_context)

        host_context.read_final_results()

        options_index < (len(options) - 1) and log_results(
            "snapshot_x"
            + str(host_context.current_options.padding.down_sampling_factor),
            host_context.oref_predicted,
            host_context.data_dimensions,
        )

        log_results(
            "probe_x" + str(host_context.current_options.padding.down_sampling_factor),
            [host_context.beam_setup.probe_refractive],
            host_context.data_dimensions,
        )

    object_update(host_context)
    print_infos(host_context)
    host_context.read_final_results()

    log_results(
        "result_x" + str(options[-1].padding.down_sampling_factor),
        host_context.oref_predicted,
        host_context.data_dimensions,
    )

    host_context.finish()

    return (
        host_context.oref_predicted[0][:, :],
        host_context.se_losses_all,
        host_context.data_dimensions.fov_size,
    )
