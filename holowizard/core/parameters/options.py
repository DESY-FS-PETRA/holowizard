import json
import sys
from .padding import Padding
from .regularization import Regularization


class Options:
    def __init__(
        self,
        update_blocks=0,
        regularization_object=Regularization(),
        regularization_probe=Regularization(),
        nesterov_object=Regularization(),
        z01_tol=0.1,
        padding=Padding(),
        verbose_interval=100,
        prototype_field=None,
    ):
        self.update_blocks = update_blocks
        self.regularization_object = regularization_object
        self.nesterov_object = nesterov_object
        self.regularization_probe = regularization_probe
        self.z01_tol = z01_tol
        self.padding = padding
        self.verbose_interval = verbose_interval
        self.prototype_field = prototype_field

    def to_log_json(self):
        return self.to_json()

    def to_json(self):
        class JsonWritable:
            def __init__(self, options: Options):
                self.update_blocks = options.update_blocks
                self.regularization_object = json.loads(options.regularization_object.to_json())
                self.nesterov_object = json.loads(options.nesterov_object.to_json())
                self.regularization_probe = json.loads(options.regularization_probe.to_json())
                self.z01_tol = options.z01_tol
                self.padding = json.loads(options.padding.to_json())
                self.verbose_interval = options.verbose_interval
                self.prototype_field = options.prototype_field

        json_writable = JsonWritable(self)

        return json.dumps(json_writable, default=lambda o: o.__dict__)

    @staticmethod
    def from_json(data):
        obj_dict = json.loads(data)
        return Options.from_dict(obj_dict)

    @staticmethod
    def from_dict(data):
        if not isinstance(data, dict):
            return None

        if (
            "update_blocks" not in data
            or "regularization_object" not in data
            or "regularization_probe" not in data
            or "z01_tol" not in data
            or "padding" not in data
            or "verbose_interval" not in data
            or "prototype_field" not in data
        ):
            return None

        padding = Padding.from_dict(data["padding"])
        regularization_object = Regularization.from_dict(data["regularization_object"])
        nesterov_object = Regularization.from_dict(data["nesterov_object"])
        regularization_probe = Regularization.from_dict(data["regularization_probe"])

        return Options(
            update_blocks=data["update_blocks"],
            regularization_object=regularization_object,
            nesterov_object=nesterov_object,
            regularization_probe=regularization_probe,
            z01_tol=data["z01_tol"],
            padding=padding,
            verbose_interval=data["verbose_interval"],
            prototype_field=data["prototype_field"],
        )

    @property
    def update_blocks(self) -> int:
        return self._update_blocks

    @property
    def regularization_object(self) -> Regularization:
        return self._regularization_object

    @property
    def nesterov_object(self) -> Regularization:
        return self._nesterov_object

    @property
    def regularization_probe(self) -> Regularization:
        return self._regularization_probe

    @property
    def z01_tol(self):
        return self._z01_tol

    @property
    def padding(self):
        return self._padding

    @property
    def verbose_interval(self):
        return self._verbose_interval

    @property
    def prototype_field(self):
        return self._prototype_field

    @update_blocks.setter
    def update_blocks(self, update_blocks) -> None:
        if isinstance(update_blocks, int):
            self._update_blocks = update_blocks
        else:
            raise TypeError(
                "Expected complex for update_blocks but got ",
                type(update_blocks),
            ).with_traceback(sys.exc_info()[2])

    @regularization_object.setter
    def regularization_object(self, regularization_object):
        if isinstance(regularization_object, Regularization):
            self._regularization_object = regularization_object
        else:
            raise TypeError(
                "Expected Regularization instance for regularization_object but got ",
                type(regularization_object),
            ).with_traceback(sys.exc_info()[2])

    @nesterov_object.setter
    def nesterov_object(self, nesterov_object):
        if isinstance(nesterov_object, Regularization):
            self._nesterov_object = nesterov_object
        else:
            raise TypeError(
                "Expected Regularization instance for nesterov_object but got ",
                type(nesterov_object),
            ).with_traceback(sys.exc_info()[2])

    @regularization_probe.setter
    def regularization_probe(self, regularization_probe):
        if isinstance(regularization_probe, Regularization):
            self._regularization_probe = regularization_probe
        else:
            raise TypeError(
                "Expected Regularization instance for regularization_probe but got ",
                type(regularization_probe),
            ).with_traceback(sys.exc_info()[2])

    @z01_tol.setter
    def z01_tol(self, z01_tol):
        if type(z01_tol) is float or type(z01_tol) is int:
            self._z01_tol = z01_tol
        else:
            raise TypeError("Expected float or int for z01_tol but got ", type(z01_tol)).with_traceback(
                sys.exc_info()[2]
            )

    @padding.setter
    def padding(self, padding):
        if isinstance(padding, Padding):
            self._padding = padding
        else:
            raise TypeError("Expected Padding instance but got ", type(padding)).with_traceback(sys.exc_info()[2])

    @verbose_interval.setter
    def verbose_interval(self, verbose_interval):
        if isinstance(verbose_interval, int) or verbose_interval is None:
            self._verbose_interval = verbose_interval
        else:
            raise TypeError("Expected integer for verbose_interval but got ", type(verbose_interval)).with_traceback(
                sys.exc_info()[2]
            )

    @prototype_field.setter
    def prototype_field(self, prototype_field):
        self._prototype_field = prototype_field
