import json
import sys
import numpy as np


class Regularization:
    def __init__(
        self,
        iterations=0,
        update_rate=0,
        gaussian_filter_fwhm=complex(0),
        values_min=-sys.float_info.max + 1j * float(0),
        values_max=float(0) + 1j * sys.float_info.max,
        l2_weight=complex(0),
        l1_weight=complex(0),
    ):
        self.iterations = iterations
        self.update_rate = update_rate
        self.gaussian_filter_fwhm = complex(gaussian_filter_fwhm)
        self.values_min = values_min
        self.values_max = values_max
        self.l2_weight = complex(l2_weight)
        self.l1_weight = complex(l1_weight)

    def to_log_json(self):
        return json.dumps(self, default=lambda o: o.__dict__)

    def to_json(self):
        class JsonWritable:
            def __init__(self, regularization: Regularization):
                self.iterations = regularization.iterations
                self.update_rate = regularization.update_rate
                self.gaussian_filter_fwhm = str(regularization.gaussian_filter_fwhm)
                self.values_min = str(regularization.values_min)
                self.values_max = str(regularization.values_max)
                self.l2_weight = str(regularization.l2_weight)
                self.l1_weight = str(regularization.l1_weight)

        json_writable = JsonWritable(self)

        return json.dumps(json_writable, default=lambda o: o.__dict__)

    @staticmethod
    def from_json(data):
        obj_dict = json.loads(data)
        return Regularization.from_dict(obj_dict)

    @staticmethod
    def from_dict(data):
        if not isinstance(data, dict):
            return None

        if (
            not "iterations" in data
            or not "update_rate" in data
            or not "gaussian_filter_fwhm" in data
            or not "values_min" in data
            or not "values_max" in data
            or not "l2_weight" in data
            or not "l1_weight" in data
        ):
            return None

        if (
            data["gaussian_filter_fwhm"] == "None"
            or data["gaussian_filter_fwhm"] is None
        ):
            gaussian_filter_fwhm = None
        else:
            gaussian_filter_fwhm = complex(data["gaussian_filter_fwhm"])

        if data["l2_weight"] == "None" or data["l2_weight"] is None:
            l2_weight = None
        else:
            l2_weight = complex(data["l2_weight"])

        if data["l1_weight"] == "None" or data["l1_weight"] is None:
            l1_weight = None
        else:
            l1_weight = complex(data["l1_weight"])

        return Regularization(
            iterations=data["iterations"],
            update_rate=data["update_rate"],
            gaussian_filter_fwhm=gaussian_filter_fwhm,
            values_min=complex(data["values_min"]),
            values_max=complex(data["values_max"]),
            l2_weight=l2_weight,
            l1_weight=l1_weight,
        )

    @property
    def iterations(self):
        return self._iterations

    @property
    def update_rate(self):
        return self._update_rate

    @property
    def gaussian_filter_fwhm(self):
        return self._gaussian_filter_fwhm

    @property
    def values_min(self):
        return self._values_min

    @property
    def values_max(self):
        return self._values_max

    @property
    def l2_weight(self):
        return self._l2_weight

    @property
    def l1_weight(self):
        return self._l1_weight

    @iterations.setter
    def iterations(self, iterations):
        if isinstance(iterations, float) or isinstance(iterations, int):
            self._iterations = iterations
        else:
            raise TypeError(
                "Expected float or int for iterations but got ",
                type(iterations),
            ).with_traceback(sys.exc_info()[2])

    @update_rate.setter
    def update_rate(self, update_rate) -> None:
        if isinstance(update_rate, float) or isinstance(update_rate, int):
            self._update_rate = update_rate
        else:
            raise TypeError(
                "Expected float or int for update_rate but got ",
                type(update_rate),
            ).with_traceback(sys.exc_info()[2])

    @gaussian_filter_fwhm.setter
    def gaussian_filter_fwhm(self, gaussian_filter_fwhm):
        if np.iscomplexobj(gaussian_filter_fwhm) or gaussian_filter_fwhm is None:
            self._gaussian_filter_fwhm = gaussian_filter_fwhm
        else:
            raise TypeError(
                "Expected complex oner None for gaussian_filter_fwhm but got ",
                type(gaussian_filter_fwhm),
            ).with_traceback(sys.exc_info()[2])

    @values_min.setter
    def values_min(self, values_min):
        if np.iscomplexobj(values_min):
            self._values_min = values_min
        else:
            raise TypeError(
                "Expected complex for values_min but got ",
                type(values_min),
            ).with_traceback(sys.exc_info()[2])

    @values_max.setter
    def values_max(self, values_max):
        if np.iscomplexobj(values_max):
            self._values_max = values_max
        else:
            raise TypeError(
                "Expected complex for values_min but got ",
                type(values_max),
            ).with_traceback(sys.exc_info()[2])

    @l2_weight.setter
    def l2_weight(self, l2_weight):
        if np.iscomplexobj(l2_weight) or l2_weight is None:
            self._l2_weight = l2_weight
        else:
            raise TypeError(
                "Expected complex for l2_weight but got ",
                type(l2_weight),
            ).with_traceback(sys.exc_info()[2])

    @l1_weight.setter
    def l1_weight(self, l1_weight):
        if np.iscomplexobj(l1_weight) or l1_weight is None:
            self._l1_weight = l1_weight
        else:
            raise TypeError(
                "Expected complex for l1_weight but got ",
                type(l1_weight),
            ).with_traceback(sys.exc_info()[2])
