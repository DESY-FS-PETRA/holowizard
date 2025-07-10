import h5py
import os
import numpy as np

from holowizard.core.utils.transform import crop_center
from holowizard.core.reconstruction.viewer.viewer import Viewer


class Hdf5Viewer(Viewer):
    def __init__(self, start_iteration=0, path=None, force=False):
        super().__init__()
        self.start_iteration = start_iteration
        self.path = path
        self.stage = 0

        if path is None:
            raise Exception("Path for intermediate results has to be defined")
        if os.path.exists(path) and force:
            os.remove(path)
        if os.path.exists(path):
            raise Exception(
                "File " + path + " already exists. Set force=True to overwrite"
            )

        self.file = h5py.File(self.path, "a")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def initialize_datasets(self, data_dimensions):
        self.file.create_group("images")

        self.file["images"].create_dataset(
            str(self.stage),
            (1, data_dimensions.fov_size[0], data_dimensions.fov_size[1]),
            dtype=np.complex64,
            chunks=True,
            maxshape=(None, data_dimensions.fov_size[0], data_dimensions.fov_size[1]),
        )

        self.file.create_group("metadata")

        self.file["metadata"].create_dataset(
            "iteration", (1,), dtype=np.uint32, chunks=True, maxshape=(None,)
        )

        self.file["metadata"].create_dataset(
            "loss", (1,), dtype=np.float32, chunks=True, maxshape=(None,)
        )

    def goto_next_stage(self, data_dimensions):
        self.stage += 1

        self.file["images"].create_dataset(
            str(self.stage),
            (1, data_dimensions.fov_size[0], data_dimensions.fov_size[1]),
            dtype=np.complex64,
            chunks=True,
            maxshape=(None, data_dimensions.fov_size[0], data_dimensions.fov_size[1]),
        )

    def stage_still_valid(self, data_dimensions):
        return (
            self.file["images"][str(self.stage)].shape[1] == data_dimensions.fov_size[0]
            and self.file["images"][str(self.stage)].shape[2]
            == data_dimensions.fov_size[1]
        )

    def update(self, iteration, object, probe, data_dimensions, loss):
        if iteration < self.start_iteration:
            return

        cropped_object = crop_center(object, data_dimensions.fov_size)

        if "images" not in self.file:
            self.initialize_datasets(data_dimensions)
        else:
            self.file["metadata"]["iteration"].resize(
                self.file["metadata"]["iteration"].shape[0] + 1, axis=0
            )
            self.file["metadata"]["loss"].resize(
                self.file["metadata"]["loss"].shape[0] + 1, axis=0
            )

            if self.stage_still_valid(data_dimensions):
                self.file["images"][str(self.stage)].resize(
                    self.file["images"][str(self.stage)].shape[0] + 1, axis=0
                )
            else:
                self.goto_next_stage(data_dimensions)

        self.file["images"][str(self.stage)][
            self.file["images"][str(self.stage)].shape[0] - 1, :, :
        ] = cropped_object.cpu().numpy()
        self.file["metadata"]["iteration"][
            self.file["metadata"]["iteration"].shape[0] - 1
        ] = iteration
        self.file["metadata"]["loss"][self.file["metadata"]["loss"].shape[0] - 1] = (
            loss[iteration].cpu().numpy()
        )
