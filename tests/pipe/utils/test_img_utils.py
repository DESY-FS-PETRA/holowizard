import os
import tempfile
import numpy as np
import tifffile

from holowizard.utils import img_utils


def create_test_images(folder, prefix, count):
    paths = []
    for i in range(count):
        path = os.path.join(folder, f"{prefix}_{i:04d}.tiff")
        tifffile.imwrite(path, np.ones((5, 5)) * i)
        paths.append(path)
    return paths


def test_extract_image_index():
    assert img_utils.extract_image_index("img_0042.tiff") == 42
    assert img_utils.extract_image_index("img_9999.tiff") == 9999
    assert img_utils.extract_image_index("not_an_image.tiff") is None


def test_extract_indices_from_list():
    files = ["img_0001.tiff", "img_0002.tiff", "img_x.tiff"]
    indices = img_utils.extract_indices_from_list(files)
    assert indices == [1, 2]


def test_find_unprocessed_image_numbers():
    all_files = ["img_0001.tiff", "img_0002.tiff", "img_0003.tiff"]
    processed = ["img_0001.tiff"]
    unprocessed = img_utils.find_unprocessed_image_numbers(all_files, processed)
    assert unprocessed == [2, 3]


def test_load_data_from_paths():
    with tempfile.TemporaryDirectory() as tmp:
        paths = create_test_images(tmp, "img", 3)
        stack = img_utils.load_data_from_paths(paths)
        assert stack.shape == (3, 5, 5)
        assert np.allclose(stack[1], 1.0)


def test_load_image():
    with tempfile.TemporaryDirectory() as tmp:
        test_file = os.path.join(tmp, "test_img.tiff")
        data = np.random.rand(10, 10)
        tifffile.imwrite(test_file, data)
        loaded = img_utils.load_image(test_file)
        assert np.allclose(data, loaded)
