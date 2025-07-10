# standard libraries
from dataclasses import dataclass

# third party libraries
from skimage.draw import disk
import numpy as np
import torch
import matplotlib.pyplot as plt

# local libraries
from holoforge.hologram import Simulation
from holoforge.hologram import Probe, BeamConfig
from holoforge.objects.shapes import Ellipse
from holoforge.factories import *
from holoforge.utils.utilities import crop_center
from holoforge.utils.unit_conversions import convert_z01, convert_z02


__all__ = [
    'forward_sim_test',
]


@dataclass
class ResultForwardSim:
    def __post_init__(self) -> None:
        self.test_names = ['Fr', 'probe', 'O_mu', 'delta', 'beta', 'obj', 'psi_exit', 'psi_det', 'holo']
        self.results = {}
        self.compared_results = {}
        
    def __eq__(self, other):
        if not isinstance(other, ResultForwardSim):
            raise ValueError(f'Cannot compare {self.__class__.__name__} with {other.__class__.__name__}.')
        
        test_Fr = np.isclose(self.results['Fr'], other.results['Fr'], atol=1e-16)
        self.compared_results['Fr'] = test_Fr
        
        test_probe = np.isclose(self.results['probe'], other.results['probe'], atol=1e-16).all()
        self.compared_results['probe'] = test_probe
        
        test_O_mu = np.isclose(self.results['O_mu'], other.results['O_mu'], atol=5e-16).all()
        self.compared_results['O_mu'] = test_O_mu
        
        test_delta = np.isclose(self.results['delta'], other.results['delta'], atol=1e-16)
        self.compared_results['delta'] = test_delta
        
        test_beta = np.isclose(self.results['beta'], other.results['beta'], atol=1e-16)
        self.compared_results['beta'] = test_beta
        
        test_obj = np.isclose(self.results['obj'], other.results['obj'], atol=1e-16).all()
        self.compared_results['obj'] = test_obj
        
        test_psi_exit = np.isclose(self.results['psi_exit'], other.results['psi_exit'], atol=1e-16).all()
        self.compared_results['psi_exit'] = test_psi_exit
        
        test_psi_det = np.isclose(self.results['psi_det'], other.results['psi_det'], atol=1e-16).all()
        self.compared_results['psi_det'] = test_psi_det
        
        test_holo = np.isclose(self.results['holo'], other.results['holo'], atol=1e-16).all()
        self.compared_results['holo'] = test_holo

    def print_last_comparison(self) -> None:
        for test_name, test_res in self.compared_results.items():
            print(f'Test "{test_name}": {"PASSED" if test_res else "FAILED"}')
        print('_' * 20)
        print(f'Total: {"PASSED" if all(self.compared_results.values()) else "FAILED"}')
    
    def append(self, name: str, result: bool) -> None:
        self.results[name] = result


def forward_sim_test(plot: bool = True) -> bool:
    results_baseline_impl = ResultForwardSim()  # code from test_simulation.py
    results_wrapped_impl = ResultForwardSim()   # own implementation wrapped in different classes
    beam_size = 4096
    holo_size = 2048
    energy = 11_000     # eV
    dx = 6500           # nm
    lam = 1.2398 / 11   # h*c / keV = nm;
    z01 = 10            # 10 * 1e7 nm = 10cm
    z02 = 20            # 20 * 1e9 nm = 20m
    thickness = 20      # 20um
    radius_disk = 128

    M = convert_z02(z02) / convert_z01(z01)
    Fr = dx**2 / (lam * (convert_z02(z02) - convert_z01(z01)) * M)
    results_baseline_impl.append('Fr', Fr)

    A = np.ones((beam_size, beam_size), dtype=np.float32)
    A_0 = 1.0 

    phi = np.ones((beam_size, beam_size), dtype=np.float32)
    phi_0 = 0.0
    probe = A_0 * A * np.exp(1j * phi_0 * phi)
    results_baseline_impl.append('probe', probe)

    transmission = 0.94781  # changed from 0.947805 to 0.94781
    O_mu = -np.log(transmission)  # Absorption coeffictient; some kind of integral over beta
    results_baseline_impl.append('O_mu', O_mu)

    delta, beta = 2.97202632E-06, 2.40418405E-08    # looked up at https://henke.lbl.gov/optical_constants/getdb2.html 
    C = delta / beta
    O_phi = -C * O_mu / 2    # phase shift (similar to absorption)
    results_baseline_impl.append('delta', delta)
    results_baseline_impl.append('beta', beta)

    O_shape = np.zeros((beam_size, beam_size))
    beam_center_px = int(beam_size / 2) - 1
    O_shape[disk((beam_center_px, beam_center_px), radius=radius_disk)] = 1
    O_ref = (O_phi + 1j * O_mu) * O_shape  
    obj = np.exp(1j * O_ref)
    results_baseline_impl.append('obj', obj)

    psi_exit = obj * probe
    results_baseline_impl.append('psi_exit', psi_exit)

    running_device = torch.device('cpu')
    sample_grid = torch.meshgrid(torch.fft.fftfreq(obj.shape[0], device=running_device, dtype=torch.float),
                                torch.fft.fftfreq(obj.shape[1], device=running_device, dtype=torch.float), indexing='ij')

    xi, eta = sample_grid
    kernel_func = torch.exp((-1j * np.pi) / Fr * (xi * xi + eta * eta)).type(torch.cfloat)
    psi_det = torch.fft.ifft2(torch.fft.fft2(torch.tensor(psi_exit, device=running_device)) * kernel_func)
    results_baseline_impl.append('psi_det', psi_det.cpu())

    holo_baseline = torch.abs(psi_det)

    holo_baseline_cropped = crop_center(holo_baseline, holo_size)
    results_baseline_impl.append('holo', holo_baseline_cropped)

    ################################## LOCAL FRAMEWORK ##############################

    shape = Ellipse(radius_disk, radius_disk)
    material = material_factory('Mg')
    phys_props = physical_properties_factory(material, energy, thickness)

    phantom = phantom_factory(shape, phys_props, size=512, center=(255, 255))
    setup = sim_setup_factory(energy, z01, z02, det_px_size=dx, phantom=phantom)

    xray = Probe(BeamConfig(), beam_size)
    results_wrapped_impl.append('probe', xray.probe)

    simulation = Simulation(setup, xray, holo_size)
    holo_wrapped_impl = simulation.forward()

    results_wrapped_impl.append('delta', phys_props.delta)
    results_wrapped_impl.append('beta', phys_props.beta)
    results_wrapped_impl.append('O_mu', -np.log(phys_props.transmission))
    results_wrapped_impl.append('Fr', setup.Fr)
    results_wrapped_impl.append('obj', simulation.obj)
    results_wrapped_impl.append('psi_exit', simulation.psi_exit)
    results_wrapped_impl.append('psi_det', simulation.psi_det)
    results_wrapped_impl.append('holo', holo_wrapped_impl)
    
    ################################## COMPARE RESULTS ###############################

    res = results_baseline_impl == results_wrapped_impl
    results_baseline_impl.print_last_comparison()

    ################################### PLOT HOLOGRAMS ###############################

    if plot:
        # Create subplots for two images
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the first image
        im1 = axes[0].imshow(holo_baseline_cropped)
        axes[0].set_title('Hologram (baseline implementation)')
        fig.colorbar(im1, ax=axes[0], orientation='vertical', pad=0.05)

        # Plot the second image
        im2 = axes[1].imshow(holo_wrapped_impl)
        axes[1].set_title('Hologram (own implementation)')
        fig.colorbar(im2, ax=axes[1], orientation='vertical', pad=0.05)
        
        # Plot the second image
        im3 = axes[2].imshow(holo_wrapped_impl - holo_baseline_cropped)
        axes[2].set_title('Abs. error (difference of both impl.)')
        fig.colorbar(im3, ax=axes[2], orientation='vertical', pad=0.05)

        # Show the plot
        plt.tight_layout()
        plt.show()

    return res
