import glob

from holowizard.core.utils.fileio import load_img_data


def default_load_data_callback(glob_data_path, image_index):
    img_files = glob.glob(glob_data_path)
    img_files.sort()

    data = load_img_data(img_files[image_index])

    return img_files[image_index], data
