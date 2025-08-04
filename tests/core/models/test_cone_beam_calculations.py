from holowizard.core.parameters.beam_setup import BeamSetup
from holowizard.core.parameters.measurement import Measurement
from holowizard.core.models.cone_beam import ConeBeam


def test_z01_calculation():
    beam_setup = BeamSetup(energy=11.0, px_size=0.0065, z02=19_661.0)
    measurement = Measurement(z01=100)

    fr_eff = ConeBeam.get_fr(beam_setup, measurement)

    z01_calculated = ConeBeam.get_z01(beam_setup, fr_eff)

    assert z01_calculated == measurement.z01
