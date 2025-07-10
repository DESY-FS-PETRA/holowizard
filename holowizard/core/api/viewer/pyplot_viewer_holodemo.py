"""
minimial viewer only showing the absorption for the hologram demonstrator
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from holowizard.core.utils.transform import crop_center

from holowizard.core.reconstruction.viewer.viewer import Viewer


class PyPlotHoloDemoViewer(Viewer):
    def __init__(self, maximize=False):
        super().__init__()
        self.fig = None
        plt.ion()
        plt.close("all")
        plt.close()
        plt.pause(0.05)
        self.fig = plt.figure(figsize=(18,13.5))

        if maximize:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()

        if matplotlib.rcParams["figure.raise_window"]:
            matplotlib.rcParams["figure.raise_window"] = False

    def update(self, iteration, object, probe, data_dimensions, loss):
        self.fig.clear()

        plot_object = crop_center(object, data_dimensions.fov_size)
        ax = self.fig.add_subplot(111)

        ax.imshow(np.rot90(plot_object.imag.cpu(), k=3), cmap="gray", interpolation="none")
        ax.set_title("Reconstructed Absorption - Live")
        ax.tick_params(left=False, bottom=False)
        ax.set_axis_off()
        m, n = (plot_object.imag.cpu()).shape

        self.fig.suptitle(f"{'Iteration '}{iteration}  Resolution {m}x{n}")
        plt.pause(0.01)
