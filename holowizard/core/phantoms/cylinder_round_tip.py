import numpy as np

from holowizard.core.phantoms.model_wave_field import model_wave_field
import holowizard.core.phantoms.options as options


def cylinder_round_tip(N):
    amp = np.zeros((N, 2 * N))

    edge_noise = N / 100
    max_radius = N / 4

    center = [int(N / 2), 2 * int(3 * N / 4)]

    for i in range(N):
        for j in range(0, 2 * N):
            x = i - center[0]
            y = j - center[1]

            max_radius_noisy = max_radius + int(edge_noise * np.random.random_sample() - edge_noise / 2)

            if j > center[1]:
                current_radius = np.linalg.norm([x, y])

                if current_radius <= max_radius_noisy:
                    height = np.sqrt(max_radius_noisy**2 - x**2 - y**2)
                    amp[i, j] = height / max_radius_noisy

            else:
                current_radius = np.abs(x)

                if current_radius <= max_radius_noisy:
                    height = np.sqrt(max_radius_noisy**2 - x**2)
                    amp[i, j] = height / max_radius_noisy
                    # amp[i,j] = 1.0

    pha = np.copy(amp)

    phantom = model_wave_field(pha, amp, options.phase_range, options.amp_range, [N, 2 * N])

    # Beamstop/Aperture
    phantom[:, 0 : N - int(N / 5)] = 0.0

    return phantom
