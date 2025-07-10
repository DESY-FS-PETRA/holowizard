import numpy
import base64
from dataclasses import dataclass


class JsonWritable:
    @dataclass
    class Array:
        raw: str
        shape: tuple
        dtype: str

    @staticmethod
    def get_array(array):
        if array is None:
            return None

        return JsonWritable.Array(
            raw=base64.b64encode(array).decode("UTF-8"),
            shape=array.shape,
            dtype=str(array.dtype),
        )

    @staticmethod
    def get_numpy_from_array(array):
        if array is None:
            return None

        raw = base64.b64decode(array["raw"].encode("UTF-8"))
        numpy_array = numpy.frombuffer(raw, array["dtype"]).reshape(array["shape"])

        return numpy_array
