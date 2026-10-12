import json

from holowizard.core.parameters import member_value_adapter
from holowizard.core.parameters.type_conversion.json_writable import JsonWritable


class FlatfieldCorrectionParams:
    def __init__(self, image, components_path):
        self.image = image
        self.components_path = components_path

    def to_log_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def to_json(self):
        class JsonInput:
            def __init__(self, flatfield_correction_params: FlatfieldCorrectionParams):
                self.components_path = flatfield_correction_params.components_path
                self.image = JsonWritable.get_array(
                    member_value_adapter.get_numpy_array_ushort(flatfield_correction_params.image)
                )

        json_writable = JsonInput(self)

        return json.dumps(json_writable, default=lambda o: o.__dict__)

    @staticmethod
    def from_json(data):
        obj_dict = json.loads(data)
        return FlatfieldCorrectionParams.from_dict(obj_dict)

    @staticmethod
    def from_dict(data):
        if not isinstance(data, dict):
            return None

        if "image" not in data or "components_path" not in data:
            return None

        image = JsonWritable.get_numpy_from_array(data["image"])

        return FlatfieldCorrectionParams(image=image, components_path=data["components_path"])

    @property
    def image(self):
        return self._image

    @property
    def components_path(self):
        return self._components_path

    @image.setter
    def image(self, value):
        self._image = value

    @components_path.setter
    def components_path(self, value):
        self._components_path = value
