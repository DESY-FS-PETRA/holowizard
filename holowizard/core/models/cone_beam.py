import logging
from holowizard.core.parameters.beam_setup import BeamSetup
from holowizard.core.parameters.measurement import Measurement


class ConeBeam(BeamSetup):
    def __init__(self):
        None

    @staticmethod
    def z12(setup: BeamSetup, measurement: Measurement):
        return (
            setup.z02 * BeamSetup.unit_z02()[1]
            - measurement.z01 * Measurement.unit_z01()[1]
        )

    @staticmethod
    def get_effective_geometry(setup: BeamSetup, measurement: Measurement):
        z12 = ConeBeam.z12(setup, measurement)
        M = (z12 + measurement.z01 * Measurement.unit_z01()[1]) / (
            measurement.z01 * Measurement.unit_z01()[1]
        )
        dx_eff = setup.px_size * BeamSetup.unit_px_size()[1] / M
        z_eff = z12 / M
        lam = 1.2398 * 1e-3 / setup.energy * BeamSetup.unit_energy()[1]
        fr_eff = dx_eff**2 / lam / z_eff

        return (
            lam,
            M,
            dx_eff / BeamSetup.unit_px_size()[1],
            z_eff / Measurement.unit_z01()[1],
            fr_eff,
        )

    @staticmethod
    def get_fr(setup: BeamSetup, measurement: Measurement, verbose=True):
        lam, M, dx_eff, z_eff, fr_eff = ConeBeam.get_effective_geometry(
            setup, measurement
        )

        if verbose is True:
            logging.info(f"{'Energy':<17}{setup.energy} " + BeamSetup.unit_energy()[0])
            logging.info(f"{'Lambda':<17}{round(lam,6)} nm")
            logging.info(f"{'Magnification':<17}{round(M,2)}")
            logging.info(
                f"{'Effective dx':<17}{round(dx_eff,3)} " + BeamSetup.unit_px_size()[0]
            )
            logging.info(
                f"{'Effective z12':<17}{round(z_eff,3)} " + Measurement.unit_z01()[0]
            )
            logging.info(f"{'Fresnel Number':<17}{fr_eff}")

        return fr_eff
