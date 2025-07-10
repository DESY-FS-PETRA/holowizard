import numpy as np

from holowizard.core.phantoms.model_wave_field import model_wave_field
import holowizard.core.phantoms.options as options


def circle(N):
    amp = np.zeros((N, N))

    center = [int(N / 2), int(N / 2)]

    for i in range(N):
        for j in range(N):
            if np.linalg.norm([i - center[0], j - center[0]]) <= N / 2:
                amp[i, j] = 1

    pha = np.copy(amp)

    phantom = model_wave_field(pha, amp, options.phase_range, options.amp_range, [N, N])

    return phantom
