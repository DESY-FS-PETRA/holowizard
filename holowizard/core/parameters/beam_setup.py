import json

from holowizard.core.parameters import member_value_adapter
from holowizard.core.parameters.type_conversion.json_writable import JsonWritable


class BeamSetup:
    def __init__(self, energy, px_size, z02, flat_field=None, probe=None):
        self.energy = energy
        self.px_size = px_size
        self.z02 = z02
        self.flat_field = flat_field
        self.probe = probe

    @staticmethod
    def unit_energy():
        return "keV", 1000

    @staticmethod
    def unit_px_size():
        return "mm", 1e6

    @staticmethod
    def unit_z02():
        return "mm", 1e6

    def to_log_json(self):
        class JsonWritable:
            def __init__(self, beam_setup: BeamSetup):
                self.energy = beam_setup.energy
                self.px_size = beam_setup.px_size
                self.z02 = beam_setup.z02

        json_writable = JsonWritable(self)

        return json.dumps(json_writable, default=lambda o: o.__dict__)

    def to_json(self):
        class JsonWriterInput:
            def __init__(self, beam_setup: BeamSetup):
                self.energy = beam_setup.energy
                self.px_size = beam_setup.px_size
                self.z02 = beam_setup.z02
                self.flat_field = JsonWritable.get_array(member_value_adapter.get_numpy_array(beam_setup.flat_field))
                self.probe = JsonWritable.get_array(beam_setup.probe)

        json_writable = JsonWriterInput(self)

        return json.dumps(json_writable, default=lambda o: o.__dict__)

    @staticmethod
    def from_json(data):
        obj_dict = json.loads(data)
        return BeamSetup.from_dict(obj_dict)

    @staticmethod
    def from_dict(data):
        if not isinstance(data, dict):
            return None

        if "energy" not in data or "px_size" not in data or "z02" not in data:
            return None

        flat_field = JsonWritable.get_numpy_from_array(data["flat_field"])
        probe = JsonWritable.get_numpy_from_array(data["probe"])

        return BeamSetup(
            energy=data["energy"],
            px_size=data["px_size"],
            z02=data["z02"],
            flat_field=flat_field,
            probe=probe,
        )

    @property
    def energy(self):
        return self._energy

    @property
    def px_size(self):
        return self._px_size

    @property
    def z02(self):
        return self._z02

    @property
    def flat_field(self):
        return self._flat_field

    @property
    def probe(self):
        return self._probe

    @property
    def probe_refractive(self):
        return member_value_adapter.get_array_refractive_from_wavefield(self._probe)

    @energy.setter
    def energy(self, energy):
        self._energy = member_value_adapter.get_float(energy)

    @px_size.setter
    def px_size(self, px_size):
        self._px_size = member_value_adapter.get_float(px_size)

    @z02.setter
    def z02(self, z02):
        self._z02 = member_value_adapter.get_float(z02)

    @flat_field.setter
    def flat_field(self, flat_field) -> None:
        self._flat_field = member_value_adapter.get_array(flat_field)

    @probe.setter
    def probe(self, probe) -> None:
        self._probe = member_value_adapter.get_array(probe)

    @probe_refractive.setter
    def probe_refractive(self, probe) -> None:
        self._probe = member_value_adapter.get_array_wavefield_from_refractive(probe)
