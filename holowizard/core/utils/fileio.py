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


def load_motor_log(scan_path):
    # Returns dict with motor names and positions
    scanname = scan_path.split("/")[-1]
    filename = scan_path + "/" + scanname + "__LogMotors.log"
    lines = [line.rstrip("\n") for line in open(filename)]
    param_names = lines[0:53]
    motor_values = lines[58].split("\t")
    motor_pos = {}
    for i, string in enumerate(param_names):
        motorname = string.split(": ")[1]
        pos = motor_values[i]
        motor_pos[motorname] = pos
    return motor_pos


def load_scan_log(scan_path):
    # TODO: needed later for tomo reco
    scanname = scan_path.split("/")[-1]
    filename = scan_path + "/" + scanname + "__LogScan.log"
    lines = [line.rstrip("\n") for line in open(filename)]
    return


def load_scan_params(scan_path):
    scanname = scan_path.split("/")[-1]
    filename = scan_path + "/" + scanname + "__ScanParam.txt"
    lines = [line.rstrip("\n") for line in open(filename)]
    scan_params = {}
    for i, line in enumerate(lines):
        if i == 11:
            scan_params[line.split(" ")[1]] = line.split(" ")[2]
        else:
            scan_params[line.split(": ")[0]] = line.split(": ")[1]
    return scan_params
