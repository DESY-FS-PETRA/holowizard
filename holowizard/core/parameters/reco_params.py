import sys
from typing import List
import json

from holowizard.core.parameters.beam_setup import BeamSetup
from holowizard.core.parameters.measurement import Measurement
from holowizard.core.parameters.options import Options
from holowizard.core.parameters.data_dimensions import DataDimensions
from holowizard.core.parameters.paths import Paths


class RecoParams:
    def __init__(
        self,
        beam_setup: BeamSetup,
        measurements: List[Measurement],
        reco_options: List[Options],
        data_dimensions: DataDimensions,
        output_path: str,
        session_params: Paths = None,
    ):
        self.beam_setup = beam_setup
        self.measurements = list(measurements)
        self.reco_options = list(reco_options)
        self.data_dimensions = data_dimensions
        self.output_path = output_path
        self.session_params = session_params

    def to_json(self):
        class JsonWritable:
            def __init__(self, reco_params: RecoParams):
                self.beam_setup = json.loads(reco_params.beam_setup.to_json())
                self.measurements = [
                    json.loads(measurement.to_json())
                    for measurement in reco_params.measurements
                ]
                self.reco_options = [
                    json.loads(reco_option.to_json())
                    for reco_option in reco_params.reco_options
                ]
                self.data_dimensions = json.loads(reco_params.data_dimensions.to_json())
                self.output_path = reco_params.output_path

        json_writable = JsonWritable(self)

        return json.dumps(json_writable, default=lambda o: o.__dict__)

    @staticmethod
    def from_json(data):
        obj_dict = json.loads(data)

        if not isinstance(obj_dict, dict):
            return None

        if (
            not "beam_setup" in obj_dict
            or not "measurements" in obj_dict
            or not "reco_options" in obj_dict
        ):
            return None

        if not "data_dimensions" in obj_dict or not "output_path" in obj_dict:
            return None

        measurements = []
        reco_options = []

        for entry in obj_dict["measurements"]:
            measurements.append(Measurement.from_dict(entry))

        for entry in obj_dict["reco_options"]:
            reco_options.append(Options.from_dict(entry))

        beam_setup = BeamSetup.from_dict(obj_dict["beam_setup"])
        data_dimensions = DataDimensions.from_dict(obj_dict["data_dimensions"])
        output_path = str(obj_dict["output_path"])

        return RecoParams(
            beam_setup=beam_setup,
            measurements=measurements,
            reco_options=reco_options,
            data_dimensions=data_dimensions,
            output_path=output_path,
        )

    @property
    def beam_setup(self) -> BeamSetup:
        return self._beam_setup

    @property
    def measurements(self) -> List[Measurement]:
        return self._measurements

    @property
    def reco_options(self) -> List[Options]:
        return self._reco_options

    @property
    def data_dimensions(self) -> DataDimensions:
        return self._data_dimensions

    @property
    def session_params(self) -> Paths:
        return self._session_params

    @property
    def output_path(self) -> str:
        return self._output_path

    @beam_setup.setter
    def beam_setup(self, beam_setup):
        if type(beam_setup) is BeamSetup:
            self._beam_setup = beam_setup
        else:
            raise TypeError(
                "Expected BeamSetup but got ", type(beam_setup)
            ).with_traceback(sys.exc_info()[2])

    @measurements.setter
    def measurements(self, measurements):
        if (
            type(measurements) is list
            and len(measurements) > 0
            and type(measurements[0]) is Measurement
        ):
            self._measurements = measurements
        else:
            raise TypeError(
                "Expected list[Measurements] but got ", type(measurements)
            ).with_traceback(sys.exc_info()[2])

    @reco_options.setter
    def reco_options(self, reco_options):
        if (
            type(reco_options) is list
            and len(reco_options) > 0
            and type(reco_options[0]) is Options
        ):
            self._reco_options = reco_options
        else:
            raise TypeError(
                "Expected list[Options] but got ", type(reco_options)
            ).with_traceback(sys.exc_info()[2])

    @data_dimensions.setter
    def data_dimensions(self, data_dimensions):
        if type(data_dimensions) is DataDimensions:
            self._data_dimensions = data_dimensions
        else:
            raise TypeError(
                "Expected DataDimensions but got ", type(data_dimensions)
            ).with_traceback(sys.exc_info()[2])

    @output_path.setter
    def output_path(self, output_path):
        if type(output_path) is str:
            self._output_path = output_path
        else:
            raise TypeError("Expected str but got ", type(output_path)).with_traceback(
                sys.exc_info()[2]
            )

    @session_params.setter
    def session_params(self, session_params):
        if isinstance(session_params, Paths) or session_params is None:
            self._session_params = session_params
        else:
            raise TypeError(
                "Expected Paths object but got ", type(session_params)
            ).with_traceback(sys.exc_info()[2])
