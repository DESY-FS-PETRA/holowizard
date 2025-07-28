import numpy as np

from holowizard.core.phantoms.model_wave_field import model_wave_field
import holowizard.core.phantoms.options as options


def triangle(N):
    amp = np.zeros((N, N))

    for i in range(N):
        amp[np.int(i) : -1, i] = 1

    pha = np.copy(amp)

    phantom = model_wave_field(pha, amp, options.phase_range, options.amp_range, [N, N])

    return phantom
