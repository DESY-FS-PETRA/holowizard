import json
from dataclasses import dataclass

from holowizard.core.parameters import member_value_adapter
from holowizard.core.parameters.type_conversion.json_writable import JsonWritable


class DataDimensions:
    def __init__(
        self,
        total_size,
        fov_size,
        window_type,
        fading_width=[(80, 80), (80, 80)],
        window=None,
    ):
        self.total_size = total_size
        self.fov_size = fov_size
        self.fov_range = [(0, fov_size[0]), (0, fov_size[1])]
        self.window_type = window_type
        self.window = window
        self.fading_width = fading_width

    def to_log_json(self):
        class JsonWritable:
            @dataclass
            class Window:
                raw: str
                shape: tuple
                dtype: str

            def __init__(self, data_dimensions: DataDimensions):
                self.total_size = data_dimensions.total_size
                self.fov_size = data_dimensions.fov_size
                self.window_type = data_dimensions.window_type
                self.fading_width = data_dimensions.fading_width

        json_writable = JsonWritable(self)

        return json.dumps(json_writable, default=lambda o: o.__dict__)

    def to_json(self):
        class JsonInput:
            def __init__(self, data_dimensions: DataDimensions):
                self.total_size = data_dimensions.total_size
                self.fov_size = data_dimensions.fov_size
                self.window_type = data_dimensions.window_type
                self.fading_width = data_dimensions.fading_width
                self.window = JsonWritable.get_array(member_value_adapter.get_numpy_array(data_dimensions.window))

        json_writable = JsonInput(self)

        return json.dumps(json_writable, default=lambda o: o.__dict__)

    @staticmethod
    def from_json(data):
        obj_dict = json.loads(data)
        return DataDimensions.from_dict(obj_dict)

    @staticmethod
    def from_dict(data):
        if not isinstance(data, dict):
            return None

        if (
            not "total_size" in data
            or not "fov_size" in data
            or not "window_type" in data
            or not "fading_width" in data
            or not "window" in data
        ):
            return None

        window = JsonWritable.get_numpy_from_array(data["window"])

        return DataDimensions(
            total_size=tuple(data["total_size"]),
            fov_size=tuple(data["fov_size"]),
            window_type=data["window_type"],
            fading_width=list(data["fading_width"]),
            window=window,
        )

    @property
    def total_size(self):
        return self._total_size

    @property
    def fov_size(self):
        return self._fov_size

    @property
    def fov_range_raw(self):
        return [
            (self._fov_range[0][0], self._fov_range[0][1]),
            (self._fov_range[1][0], self._fov_range[1][1]),
        ]

    @property
    def fov_range(self):
        return [
            slice(self._fov_range[0][0], self._fov_range[0][1]),
            slice(self._fov_range[1][0], self._fov_range[1][1]),
        ]

    @property
    def fading_width(self):
        return self._fading_width

    @property
    def window_type(self):
        return self._window_type

    @property
    def window(self):
        return self._window

    @total_size.setter
    def total_size(self, total_size):
        self._total_size = member_value_adapter.get_tuple(total_size)

    @fov_size.setter
    def fov_size(self, fov_size):
        self._fov_size = member_value_adapter.get_tuple(fov_size)

    @fov_range.setter
    def fov_range(self, fov_range) -> None:
        self._fov_range = member_value_adapter.get_range_tuple_list(fov_range)

    @window_type.setter
    def window_type(self, window_type) -> None:
        self._window_type = member_value_adapter.get_string(window_type)

    @window.setter
    def window(self, window):
        self._window = member_value_adapter.get_array_float(window)

    @fading_width.setter
    def fading_width(self, fading_width) -> None:
        self._fading_width = member_value_adapter.get_range_tuple_list(fading_width)
