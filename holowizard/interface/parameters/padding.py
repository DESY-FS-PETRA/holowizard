import json
import sys
from enum import Enum
import numpy


class Padding:
    class PaddingMode(Enum):
        UNDEFINED = 0
        CONSTANT = 1
        MIRROR_LEFT = 2
        MIRROR_HORIZONTAL = 3
        MIRROR_ALL = 4
        REPETITION_LEFT = 5
        REPETITION_ALL = 6
        REPETITION_HORIZONTAL = 7

    def __init__(
        self,
        padding_mode=PaddingMode.CONSTANT,
        padding_factor=2,
        down_sampling_factor=1,
        cutting_band=0,
        a0=1.0,
        prototype_field=None,
    ):
        self.padding_mode = padding_mode
        self.padding_factor = padding_factor
        self.down_sampling_factor = down_sampling_factor
        self.cutting_band = cutting_band
        self.a0 = a0
        self.prototype_field = prototype_field

    def to_log_json(self):
        return json.dumps(self, default=lambda o: o.__dict__)

    def to_json(self):
        class JsonWritable:
            def __init__(self, padding: Padding):
                self.padding_mode = padding.padding_mode.name
                self.padding_factor = padding.padding_factor
                self.down_sampling_factor = padding.down_sampling_factor
                self.cutting_band = padding.cutting_band
                self.a0 = padding.a0
                self.prototype_field = padding.prototype_field

        json_writable = JsonWritable(self)

        return json.dumps(json_writable, default=lambda o: o.__dict__)

    @staticmethod
    def from_json(data):
        obj_dict = json.loads(data)
        return Padding.from_dict(obj_dict)

    @staticmethod
    def from_dict(data):

        if not isinstance(data, dict):
            return None

        if (
            not "padding_mode" in data
            or not "padding_factor" in data
            or not "down_sampling_factor" in data
        ):
            return None

        if (
            not "cutting_band" in data
            or not "a0" in data
            or not "prototype_field" in data
        ):
            return None

        padding_mode = Padding.PaddingMode[data["padding_mode"]]

        return Padding(
            padding_mode=padding_mode,
            padding_factor=data["padding_factor"],
            down_sampling_factor=data["down_sampling_factor"],
            cutting_band=data["cutting_band"],
            a0=data["a0"],
            prototype_field=data["prototype_field"],
        )

    @property
    def padding_mode(self):
        return Padding.PaddingMode[self._padding_mode]

    @property
    def padding_factor(self):
        return self._padding_factor

    @property
    def down_sampling_factor(self):
        return self._down_sampling_factor

    @property
    def cutting_band(self):
        return self._cutting_band

    @property
    def a0(self):
        return self._a0

    @property
    def prototype_field(self):
        return self._prototype_field

    @padding_mode.setter
    def padding_mode(self, padding_mode):
        if type(padding_mode) is Padding.PaddingMode:
            self._padding_mode = padding_mode.name
        elif type(padding_mode) is int:
            self._padding_mode = Padding.PaddingMode(padding_mode).name
        elif type(padding_mode) is str:
            self._padding_mode = Padding.PaddingMode[padding_mode].name
        else:
            raise TypeError(
                "Expected PaddingMode or int but got ", type(padding_mode)
            ).with_traceback(sys.exc_info()[2])

    @padding_factor.setter
    def padding_factor(self, padding_factor):
        if type(padding_factor) is int or type(padding_factor) is float:
            self._padding_factor = padding_factor
        else:
            raise TypeError(
                "Expected int or float but got ", type(padding_factor)
            ).with_traceback(sys.exc_info()[2])

    @down_sampling_factor.setter
    def down_sampling_factor(self, down_sampling_factor):
        if type(down_sampling_factor) is int:
            self._down_sampling_factor = down_sampling_factor
        else:
            raise TypeError(
                "Expected int but got ", type(down_sampling_factor)
            ).with_traceback(sys.exc_info()[2])

    @cutting_band.setter
    def cutting_band(self, cutting_band):
        if type(cutting_band) is int:
            self._cutting_band = cutting_band
        else:
            raise TypeError("Expected int but got ", type(cutting_band)).with_traceback(
                sys.exc_info()[2]
            )

    @a0.setter
    def a0(self, a0):
        if isinstance(a0, float) or isinstance(a0, numpy.float32):
            self._a0 = float(a0)
        else:
            raise TypeError("Expected int or float but got ", type(a0)).with_traceback(
                sys.exc_info()[2]
            )

    @prototype_field.setter
    def prototype_field(self, prototype_field):
        self._prototype_field = prototype_field
