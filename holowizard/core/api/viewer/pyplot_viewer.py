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

        axs0 = self.fig.add_subplot(221)
        axs1 = self.fig.add_subplot(222)

        axs2 = self.fig.add_subplot(223)
        axs3 = self.fig.add_subplot(224)

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

        axs3.plot(loss[0 : (iteration + 1)].cpu())
        axs3.set_yscale("log")
        axs3.set_title("MSE")

        self.fig.suptitle(f"{'Iteration '}{iteration}")
        plt.pause(0.01)
