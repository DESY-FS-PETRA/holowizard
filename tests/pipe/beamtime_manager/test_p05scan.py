import os
import tempfile
import numpy as np
import tifffile

from holowizard.beamtime.P05 import P05Scan

def create_dummy_scan_folder(base_path, scan_name):
    raw_path = os.path.join(base_path, "raw", scan_name)
    os.makedirs(raw_path, exist_ok=True)

    # Create fake images
    for i in range(3):
        tifffile.imwrite(os.path.join(raw_path, f"img_{i:04d}.tif"), np.ones((10, 10)) * i)

    # Create a dummy ScanParam.txt
    with open(os.path.join(raw_path, f"{scan_name}__ScanParam.txt"), 'w') as f:
        f.write("energy: 11.0\nholder: 195.0")

    return raw_path

def test_scan_initialization_and_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        scan_name = "nano_test"
        beamtime = "BT123"
        year = "2025"
        base_path = os.path.join(tmp, year, "data", beamtime)

        scan_path = create_dummy_scan_folder(base_path, scan_name)

        scan = P05Scan(scan_name, beamtime, year)
        scan.path_raw = scan_path  # override for test
        scan.meta_dict = scan.get_metadata()

        assert scan.name == scan_name
        assert scan.beamtime == beamtime
        assert "energy" in scan.meta_dict

def test_load_image_success_and_fallback():
    with tempfile.TemporaryDirectory() as tmp:
        scan_name = "scan_img_test"
        base = create_dummy_scan_folder(tmp, scan_name)

        scan = P05Scan(scan_name, "BT", "2025")
        scan.path_raw = base
        img = scan.load_image(0)
        assert img.shape == (10, 10)

        # Force fallback
        scan.num_img = 0
        img = scan.load_image(999)
        assert img.shape == (100, 100)  # fallback shape

def test_set_scan_setup():
    scan = P05Scan("demo", "bt", "2025")
    scan.set_scan_setup(energy=11.0, holder=200.0)

    assert scan.energy == 11.0
    assert scan.holder_length == 200.0

