import os
import numpy as np
from skimage import io

from holowizard.core.phantoms.model_wave_field import model_wave_field
import holowizard.core.phantoms.options as options


def dicty_sketch(N):
    this_file_path = os.path.dirname(os.path.realpath(__file__))
    this_dir_path, this_filename = os.path.split(this_file_path)

    amp = io.imread(this_dir_path + "/phantoms/dicty_sketch.png")
    pha = np.copy(amp)

    phantom = model_wave_field(pha, amp, options.phase_range, options.amp_range, [N, N])

    return phantom
