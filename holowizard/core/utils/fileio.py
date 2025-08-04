from skimage import io

from multiprocessing.dummy import Pool
import numpy as np
from tqdm import tqdm
from itertools import repeat


def load_img_data(img_file, progress_bar=None):
    img_data = io.imread(img_file)
    if progress_bar is not None:
        progress_bar.update(1)
    return img_data


def load_multi_img_data(img_files):
    print("Reading ", len(img_files), " images")
    pool = Pool()
    with tqdm(total=len(img_files)) as pbar:
        img_data = list(pool.starmap(load_img_data, zip(img_files, repeat(pbar))))

    return img_data


def write_img_data(img_file, img_data):
    io.imsave(img_file, np.float32(img_data))