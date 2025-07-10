import torch
import numpy
from dataclasses import dataclass
import base64


class SerializableArray:
    @dataclass
    class Data:
        raw: str
        shape: tuple
        dtype: str

    def __init__(self, data_array):
        if type(data_array) is torch.Tensor:
            data_array = numpy.ascontiguousarray(data_array.cpu().numpy())
        else:
            data_array = numpy.ascontiguousarray(data_array)

        self.data = SerializableArray.Data(
            base64.b64encode(data_array).decode("UTF-8"),
            data_array.shape,
            str(data_array.dtype),
        )

    def get_dict(self):
        return self.data.__dict__

    @staticmethod
    def numpy_from_dict(dict):
        data_raw = base64.b64decode(dict["raw"].encode("UTF-8"))
        data_numpy = numpy.frombuffer(data_raw, dict["dtype"]).reshape(dict["shape"])
        return data_numpy
