import os
import tempfile
import numpy as np
import tifffile
import pytest
from holowizard.utils.project_utils import return_a0, get_images_list, calculate_image_statistics, format_result


def create_dummy_images(directory, prefix, count, value_fn):
    paths = []
    for i in range(count):
        img = np.ones((5, 5)) * value_fn(i)
        path = os.path.join(directory, f"{prefix}_{i:04d}.tiff")
        tifffile.imwrite(path, img)
        paths.append(path)
    return paths


def test_get_images_list_and_return_a0():
    with tempfile.TemporaryDirectory() as tmp:
        img_paths = create_dummy_images(tmp, "img", 3, lambda i: i + 1)
        ref_paths = create_dummy_images(tmp, "ref", 3, lambda i: 2 * (i + 1))

        images, refs = get_images_list(tmp)
        assert len(images) == 3
        assert len(refs) == 3

        a0 = return_a0(tmp)
        assert isinstance(a0, float)
        assert a0 > 0


def test_return_a0_handles_missing_files():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            return_a0(tmp)


def test_calculate_image_statistics():
    with tempfile.TemporaryDirectory() as tmp:
        test_path = os.path.join(tmp, "test_img.tiff")
        image = np.array([[1, 2], [3, 4]])
        tifffile.imwrite(test_path, image)

        stats = calculate_image_statistics(test_path)
        assert np.isclose(stats["mean"], 2.5)
        assert np.isclose(stats["min"], 1.0)
        assert np.isclose(stats["max"], 4.0)


def test_format_result():
    formatted = format_result("Energy", 11.0, "keV")
    assert formatted == "Energy: 11.0 keV"
    formatted = format_result("Holder", 195.0)
    assert formatted == "Holder: 195.0"
