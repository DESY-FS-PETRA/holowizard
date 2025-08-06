import matplotlib
import matplotlib.pyplot as plt
from holowizard.core.utils.transform import crop_center

from holowizard.core.reconstruction.viewer.viewer import Viewer


class PyPlotViewer(Viewer):
    def __init__(self, maximize=False):
        super().__init__()
        self.fig = None
        plt.ion()
        plt.close("all")
        plt.close()
        plt.pause(0.05)
        self.fig = plt.figure(figsize=(12.8, 8.8))

        if maximize:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()

        if matplotlib.rcParams["figure.raise_window"]:
            matplotlib.rcParams["figure.raise_window"] = False

    def update(self, iteration, object, probe, data_dimensions, loss):
        self.fig.clear()

        axs0 = self.fig.add_subplot(321)
        axs1 = self.fig.add_subplot(322)

        axs2 = self.fig.add_subplot(325)
        axs3 = self.fig.add_subplot(326)

        axs4 = self.fig.add_subplot(323)
        axs5 = self.fig.add_subplot(324)

        plot_object = crop_center(object, data_dimensions.fov_size)
        crossect_pos = int(plot_object.shape[1] / 2)

        pos = axs0.imshow(plot_object.real.cpu(), cmap="gray", interpolation="none")
        self.fig.colorbar(pos, ax=axs0, fraction=0.046, pad=0.04)
        axs0.set_title("Object Real")
        axs0.axhline(crossect_pos, linestyle="--")
        axs0.tick_params(left=False, bottom=False)

        pos = axs1.imshow(plot_object.imag.cpu(), cmap="gray_r", interpolation="none")
        self.fig.colorbar(pos, ax=axs1, fraction=0.046, pad=0.04)
        axs1.set_title("Object Imag")
        axs1.tick_params(left=False, bottom=False)

        axs2.plot(plot_object[crossect_pos, :].real.cpu())
        axs2.set_title("Object Real [" + str(crossect_pos) + ",:]")

        axs3.plot(loss[0:iteration].cpu())
        axs3.set_yscale("log")
        axs3.set_title("MSE")

        plot_probe = crop_center(probe, data_dimensions.fov_size)

        pos = axs4.imshow(plot_probe.real.cpu(), cmap="gray", interpolation="none")
        self.fig.colorbar(pos, ax=axs4, fraction=0.046, pad=0.04)
        axs4.set_title("Probe Real")
        axs4.tick_params(left=False, bottom=False)

        pos = axs5.imshow(plot_probe.imag.cpu(), cmap="gray_r", interpolation="none")
        self.fig.colorbar(pos, ax=axs5, fraction=0.046, pad=0.04)
        axs5.set_title("Probe Imag")
        axs5.tick_params(left=False, bottom=False)

        self.fig.suptitle(f"{'Iteration '}{iteration}")
        plt.pause(0.01)
