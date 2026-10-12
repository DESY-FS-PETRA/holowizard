import json

from holowizard.core.src.src.parameter_classes import member_value_adapter
from holowizard.core.src.parameter_classes.type_conversion.json_writable import JsonWritable


class Measurement:
    def __init__(self, z01, angle=0, data_path="", data=None, z01_confidence=5.0):
        self.z01 = z01
        self.angle = angle
        self.z01_confidence = z01_confidence
        self.data = data
        self.data_path = data_path

    @staticmethod
    def unit_z01():
        return "mm", 1e6

    def to_log_json(self):
        class JsonInput:
            def __init__(self, measurement: Measurement):
                self.z01 = measurement.z01
                self.z01_confidence = measurement.z01_confidence
                self.angle = measurement.angle
                self.data_path = member_value_adapter.get_string(measurement.data_path)

        json_writable = JsonInput(self)

        return json.dumps(json_writable, default=lambda o: o.__dict__)

    def to_json(self):
        class JsonInput:
            def __init__(self, measurement: Measurement):
                self.z01 = measurement.z01
                self.angle = measurement.angle
                self.z01_confidence = measurement.z01_confidence
                self.data_path = str(measurement.data_path)
                self.data = JsonWritable.get_array(member_value_adapter.get_numpy_array_float(measurement.data))

        json_writable = JsonInput(self)

        return json.dumps(json_writable, default=lambda o: o.__dict__)

    @staticmethod
    def from_json(data):
        obj_dict = json.loads(data)
        return Measurement.from_dict(obj_dict)

    @staticmethod
    def from_dict(data):
        if not isinstance(data, dict):
            return None

        if (
            "z01" not in data
            or "angle" not in data
            or "z01_confidence" not in data
            or "data_path" not in data
            or "data" not in data
        ):
            return None

        hologram = JsonWritable.get_numpy_from_array(data["data"])

        return Measurement(
            z01=data["z01"],
            angle=data["angle"],
            z01_confidence=data["z01_confidence"],
            data_path=data["data_path"],
            data=hologram,
        )

    @property
    def data(self):
        return self._data

    @property
    def data_path(self):
        return self._data_path

    @property
    def z01(self):
        return self._z01

    @property
    def angle(self):
        return self._angle

    @property
    def z01_confidence(self):
        return self._z01_confidence

    @property
    def z01_bounds(self):
        return (self.z01 - self.z01_confidence, self.z01 + self.z01_confidence)

    @data.setter
    def data(self, data) -> None:
        self._data = member_value_adapter.get_array_float(data)

    @data_path.setter
    def data_path(self, data_path) -> None:
        self._data_path = member_value_adapter.get_path(data_path)

    @z01.setter
    def z01(self, z01) -> None:
        self._z01 = member_value_adapter.get_float(z01)

    @angle.setter
    def angle(self, angle) -> None:
        self._angle = member_value_adapter.get_float(angle)

    @z01_confidence.setter
    def z01_confidence(self, z01_confidence):
        self._z01_confidence = member_value_adapter.get_float(z01_confidence)
