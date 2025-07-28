import json
import sys

from holowizard.core.parameters.type_conversion.json_writable import JsonWritable
from holowizard.core.parameters import member_value_adapter


class FlatfieldComponentsParams:
    def __init__(self, measurements, num_components, save_path):
        self.measurements = measurements
        self.num_components = num_components
        self.save_path = save_path

    def to_log_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def to_json(self):
        class JsonInput:
            def __init__(self, flatfield_components_params: FlatfieldComponentsParams):
                self.measurements = [
                    JsonWritable.get_array(member_value_adapter.get_numpy_array_ushort(measurement))
                    for measurement in flatfield_components_params.measurements
                ]
                self.num_components = flatfield_components_params.num_components
                self.save_path = flatfield_components_params.save_path

        json_writable = JsonInput(self)

        return json.dumps(json_writable, default=lambda o: o.__dict__)

    @staticmethod
    def from_json(data):
        obj_dict = json.loads(data)
        return FlatfieldComponentsParams.from_dict(obj_dict)

    @staticmethod
    def from_dict(data):
        if not isinstance(data, dict):
            return None

        if not "measurements" in data or not "num_components" in data or not "save_path" in data:
            return None

        measurements = []
        measurements_dict = data["measurements"]

        for measurement in measurements_dict:
            measurements.append(JsonWritable.get_numpy_from_array(measurement))

        return FlatfieldComponentsParams(
            measurements=measurements,
            num_components=data["num_components"],
            save_path=data["save_path"],
        )

    @property
    def measurements(self):
        return self._measurements

    @property
    def num_components(self):
        return self._num_components

    @property
    def save_path(self):
        return self._save_path

    @measurements.setter
    def measurements(self, value):
        if type(value) is list:
            self._measurements = value
        else:
            raise TypeError("Expected list for measurements but got ", type(value)).with_traceback(sys.exc_info()[2])

    @num_components.setter
    def num_components(self, value):
        self._num_components = value

    @save_path.setter
    def save_path(self, value):
        self._save_path = value
