import numpy as np

from holowizard.core.phantoms.model_wave_field import model_wave_field
import holowizard.core.phantoms.options as options


def edge(N):
    amp = np.zeros((N, N))

    amp[0:-1, 0 : int(N / 2)] = 1

    pha = np.copy(amp)

    phantom = model_wave_field(pha, amp, options.phase_range, options.amp_range, [N, N])

    return phantom
