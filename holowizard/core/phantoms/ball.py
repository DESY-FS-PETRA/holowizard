import numpy as np

from holowizard.core.phantoms.model_wave_field import model_wave_field
import holowizard.core.phantoms.options as options


def ball(N):
    amp = np.zeros((N, N))

    max_radius = N / 2

    center = [int(N / 2), int(N / 2), int(N / 2)]

    for i in range(N):
        for j in range(N):
            x = i - center[0]
            y = j - center[1]

            current_radius = np.linalg.norm([x, y])

            if current_radius <= max_radius:
                height = np.sqrt(max_radius**2 - x**2 - y**2)
                amp[i, j] = height / max_radius

    pha = np.copy(amp)

    phantom = model_wave_field(pha, amp, options.phase_range, options.amp_range, [N, N])

    return phantom
