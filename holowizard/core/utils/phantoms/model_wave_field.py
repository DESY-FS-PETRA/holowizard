from skimage import transform
import numpy as np


def model_wave_field(pha, amp, phaseRange, amplitudeRange, imageSize):
    # this function takes 2 arrays as arguments which are interpreted as
    # amplitude and phase of a wavefield(probe or object or whatever). The user
    # can set the ranges of amplitude and phase. p is a parameter struture, its
    # purpose here is to give the size of the desired wavefield.

    # HoloTomoToolbox
    # Copyright (C) 2019  Institut fuer Roentgenphysik, Universitaet Goettingen
    #
    # This program is free software: you can redistribute it and/or modify
    # it under the terms of the GNU General Public License as published by
    # the Free Software Foundation, either version 3 of the License, or
    # (at your option) any later version.
    #
    # This program is distributed in the hope that it will be useful,
    # but WITHOUT ANY WARRANTY; without even the implied warranty of
    # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    # GNU General Public License for more details.
    #
    # You should have received a copy of the GNU General Public License
    # along with this program.  If not, see <http://www.gnu.org/licenses/>.
    #
    # Original author: Johannes Hagemann
    ##

    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    lowerPhase = phaseRange[0]
    upperPhase = phaseRange[1]
    lowerAmplitude = amplitudeRange[0]
    upperAmplitude = amplitudeRange[1]

    # convert only to grayscale if we have a color image, otherwise rgb2gray
    # throws an error
    if len(amp.shape) > 2:
        amp = rgb2gray(amp)

    if len(pha.shape) > 2:
        pha = rgb2gray(pha)

    amp = transform.resize(amp, imageSize, order=3)
    pha = transform.resize(pha, imageSize, order=3)

    # scaled amplitudes
    ampRange = np.max(amp) - np.min(amp)
    amp = lowerAmplitude + ((amp) / (ampRange)) * (upperAmplitude - lowerAmplitude)

    # scaled phases
    phaRange = np.max(pha) - np.min(pha)
    pha = lowerPhase + ((pha) / (phaRange)) * (upperPhase - lowerPhase)

    beam = np.exp(-amp) * np.exp(-1j * pha)
    return beam
