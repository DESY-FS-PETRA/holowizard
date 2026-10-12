import json
import torch
from dataclasses import dataclass
from typing import List
from pathlib import Path

import holowizard.core
from holowizard.core.parameters.beam_setup import BeamSetup
from holowizard.core.parameters.measurement import Measurement
from holowizard.core.parameters.options import Options
from holowizard.core.parameters.data_dimensions import DataDimensions
from holowizard.core.parameters import member_value_adapter, measurement
from holowizard.core.parameters.type_conversion.json_writable import JsonWritable
from holowizard.core.reconstruction.viewer import Viewer


@dataclass(init=True)
class ContextProjectionSingleStage:
    key: str = ""
    viewer: List[Viewer] = None

    options: Options = None
    oref_predicted: torch.Tensor = None
    nesterov_vt: torch.Tensor = None
    grad_probe: torch.Tensor = None
    fresnel_numbers: List[float] = None
    measurements: List[Measurement] = None  # A list, because of multidistance measurements
    data_dimensions: DataDimensions = None
    beam_setup: BeamSetup = None

    se_losses_all: torch.Tensor = None
    current_iter_offset: int = 0

    absorption_min: torch.Tensor = None
    absorption_max: torch.Tensor = None
    phaseshift_min: torch.Tensor = None
    phaseshift_max: torch.Tensor = None

    oref_predicted_path: Path = None

    def to_json_writable(self):
        class JsonInput:
            @dataclass
            class Data:
                raw: str
                shape: tuple
                dtype: str

            def __init__(self, context: ContextProjectionSingleStage):
                self.key = context.key
                self.viewer = None
                self.options = context.options.to_json()
                self.oref_predicted = JsonWritable.get_array(
                    member_value_adapter.get_numpy_array_complex(context.oref_predicted)
                )
                self.nesterov_vt = JsonWritable.get_array(
                    member_value_adapter.get_numpy_array_complex(context.nesterov_vt)
                )
                self.grad_probe = JsonWritable.get_array(
                    member_value_adapter.get_numpy_array_complex(context.grad_probe)
                )
                self.fresnel_numbers = context.fresnel_numbers
                self.measurements = [json.loads(measurement.to_json()) for measurement in context.measurements]
                self.data_dimensions = context.data_dimensions.to_json()

                self.beam_setup = context. beam_setup.to_json()

                self.se_losses_all = JsonWritable.get_array(
                    member_value_adapter.get_numpy_array_float(context.se_losses_all)
                )

                self.current_iter_offset = context.current_iter_offset

                self.oref_predicted_path = member_value_adapter.get_string(context.oref_predicted_path)

                self.absorption_min = context.absorption_min.item()
                self.absorption_max = context.absorption_max.item()
                self.phaseshift_min = context.phaseshift_min.item()
                self.phaseshift_max = context.phaseshift_max.item()

        json_writable = JsonInput(self)

        return json_writable

    def to_json(self):
        return json.dumps(self.to_json_writable(), default=lambda o: o.__dict__)

    @staticmethod
    def from_dict(data, device=None):
        if device is None:
            device = holowizard.core.torch_running_device

        if not isinstance(data, dict):
            return None

        if (
            "key" not in data
            or "viewer" not in data
            or "options" not in data
            or "oref_predicted" not in data
            or "nesterov_vt" not in data
            or "grad_probe" not in data
            or "fresnel_numbers" not in data
            or "measurements" not in data
            or "data_dimensions" not in data
            or "beam_setup" not in data
            or "se_losses_all" not in data
            or "current_iter_offset" not in data
            or "absorption_min" not in data
            or "absorption_max" not in data
            or "phaseshift_min" not in data
            or "phaseshift_max" not in data
        ):
            return None

        key = data["key"]
        viewer = data["viewer"]
        options = Options.from_json(data["options"])

        if data["oref_predicted"] is None:
            oref_predicted = None
        if data["nesterov_vt"] is not None:
            nesterov_vt = torch.tensor(JsonWritable.get_numpy_from_array(data["nesterov_vt"]), device=device)
        else:
            nesterov_vt = None
        if data["grad_probe"] is not None:
            grad_probe = torch.tensor(JsonWritable.get_numpy_from_array(data["grad_probe"]), device=device)
        else:
            grad_probe = None

        fresnel_numbers = data["fresnel_numbers"]

        measurements = []
        for entry in data["measurements"]:
            measurements.append(Measurement.from_dict(entry))
            if measurements[-1].data is not None:
                measurements[-1].data = measurements[-1].data.to(device)

        data_dimensions = DataDimensions.from_json(data["data_dimensions"])
        beam_setup = BeamSetup.from_json(data["beam_setup"])
        if beam_setup.probe_refractive is not None:
            beam_setup.probe_refractive = beam_setup.probe_refractive.to(device)
        se_losses_all = torch.tensor(JsonWritable.get_numpy_from_array(data["se_losses_all"]), device=device)
        current_iter_offset = data["current_iter_offset"]
        oref_predicted_path = member_value_adapter.get_path(data["oref_predicted_path"])
        absorption_min = torch.tensor(data["absorption_min"], device=device)
        absorption_max = torch.tensor(data["absorption_max"], device=device)
        phaseshift_min = torch.tensor(data["phaseshift_min"], device=device)
        phaseshift_max = torch.tensor(data["phaseshift_max"], device=device)

        return ContextProjectionSingleStage(
            key=key,
            viewer=viewer,
            options=options,
            beam_setup=beam_setup,
            measurements=measurements,
            data_dimensions=data_dimensions,
            fresnel_numbers=fresnel_numbers,
            #oref_predicted=oref_predicted,
            nesterov_vt=nesterov_vt,
            grad_probe=grad_probe,
            se_losses_all=se_losses_all,
            current_iter_offset=current_iter_offset,
            absorption_min=absorption_min,
            absorption_max=absorption_max,
            phaseshift_min=phaseshift_min,
            phaseshift_max=phaseshift_max,
            oref_predicted_path=oref_predicted_path
        )

    @staticmethod
    def from_json(data, device=None):
        if device is None:
            device = holowizard.core.torch_running_device
        obj_dict = json.loads(data)
        return ContextProjectionSingleStage.from_dict(obj_dict, device)
