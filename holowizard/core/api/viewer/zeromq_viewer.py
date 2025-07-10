import zmq
import skimage
import json
import numpy as np
from holowizard.core.utils.transform import crop_center
from holowizard.core.reconstruction.viewer.viewer import Viewer


class ZeroMQViewer(Viewer):
    def __init__(self, port):
        super().__init__()
        self.reco_context = zmq.Context()
        self.reco_socket = self.reco_context.socket(zmq.PUB)
        self.reco_socket.bind("tcp://*:" + str(port))

    def update(self, iteration, object, probe, data_dimensions, loss):
        object_cropped = crop_center(object, data_dimensions.fov_size)
        probe_cropped = crop_center(probe, data_dimensions.fov_size)

        datasources = [
            "ObjectReal",
            "ObjectRealCropped",
            "ObjectImag",
            "ObjectImagCropped",
            "ProbeReal",
            "ProbeRealCropped",
            "ProbeImag",
            "ProbeImagCropped",
        ]
        data = [
            object.real,
            object_cropped.real,
            object.imag,
            object_cropped.imag,
            probe.real,
            probe_cropped.real,
            probe.imag,
            probe_cropped.imag,
        ]

        metadata = {"datasources:": datasources}

        message = (
            "datasources".encode("ascii", "ignore"),
            json.dumps(metadata).encode("ascii", "ignore"),
            "JSON".encode("ascii", "ignore"),
        )

        self.reco_socket.send_multipart(message)

        for i in range(len(datasources)):
            tfilter = datasources[i]
            value = data[i].cpu().numpy()
            shape = value.shape
            dtype = value.dtype.name
            imagename = datasources[i] + "_snapshot_%s" % iteration
            message = (
                tfilter.encode("ascii", "ignore"),
                np.ascontiguousarray(value),
                json.dumps(shape).encode("ascii", "ignore"),
                json.dumps(dtype).encode("ascii", "ignore"),
                imagename.encode("ascii", "ignore"),
                "JSON".encode("ascii", "ignore"),
            )

            self.reco_socket.send_multipart(message)
