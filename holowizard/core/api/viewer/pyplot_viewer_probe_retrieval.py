import matplotlib
import matplotlib.pyplot as plt
from holowizard.core.src.misc.transform import crop_center

from holowizard.core.src.visualization.viewer import Viewer


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

    def update(self, iteration, hologram_measured, hologram_guess, object, probe, data_dimensions, loss):
        self.fig.clear()

        axs0 = self.fig.add_subplot(331)
        axs1 = self.fig.add_subplot(332)
        axs2 = self.fig.add_subplot(333)

        axs3 = self.fig.add_subplot(334)
        axs4 = self.fig.add_subplot(335)
        axs5 = self.fig.add_subplot(336)

        axs6 = self.fig.add_subplot(337)
        axs7 = self.fig.add_subplot(338)
        axs8 = self.fig.add_subplot(339)

        if hologram_measured is not None:
            plot_hologram = crop_center(hologram_measured, data_dimensions.fov_size)
            crossect_pos = int(plot_hologram.shape[1] / 2)
            pos = axs0.imshow(plot_hologram.cpu(), cmap="gray", interpolation="none")
            self.fig.colorbar(pos, ax=axs0, fraction=0.046, pad=0.04)
            axs0.set_title("Measured Hologram")
            axs0.axhline(crossect_pos, linestyle="--", color='red')
            axs0.tick_params(left=False, bottom=False)
            axs6.plot(plot_hologram[crossect_pos, :].cpu(),color='red')
        else:
            axs0.set_axis_off()

        if hologram_guess is not None:
            plot_hologram = crop_center(hologram_guess, data_dimensions.fov_size)
            crossect_pos = int(plot_hologram.shape[1] / 2)
            pos = axs3.imshow(plot_hologram.cpu(), cmap="gray", interpolation="none")
            self.fig.colorbar(pos, ax=axs3, fraction=0.046, pad=0.04)
            axs3.set_title("Propagated Probe")
            axs3.axhline(crossect_pos, linestyle="--", color='blue')
            axs3.tick_params(left=False, bottom=False)

            axs6.plot(plot_hologram[crossect_pos, :].cpu(),color='blue')
            axs6.set_title("Propagated Probe vs Hologram [" + str(crossect_pos) + ",:]")
                
        else:
            axs3.set_axis_off()
            if hologram_measured is None:
                axs6.set_axis_off()

        if object is not None:
            plot_object = crop_center(object, data_dimensions.fov_size)

            crossect_pos = int(plot_object.shape[1] / 2)

            pos = axs1.imshow(plot_object.real.cpu(), cmap="gray", interpolation="none")
            self.fig.colorbar(pos, ax=axs1, fraction=0.046, pad=0.04)
            axs1.set_title("Object Real")
            axs1.axhline(crossect_pos, linestyle="--")
            axs1.tick_params(left=False, bottom=False)

            pos = axs2.imshow(plot_object.imag.cpu(), cmap="gray_r", interpolation="none")
            self.fig.colorbar(pos, ax=axs2, fraction=0.046, pad=0.04)
            axs2.set_title("Object Imag")
            axs2.tick_params(left=False, bottom=False)

            axs7.plot(plot_object[crossect_pos, :].real.cpu())
            axs7.set_title("Object Real [" + str(crossect_pos) + ",:]")
        else:
            axs1.set_axis_off()
            axs2.set_axis_off()
            axs7.set_axis_off()

        axs8.plot(loss[0:iteration].cpu())
        axs8.set_yscale("log")
        axs8.set_title("MSE")

        if probe is not None:
            plot_probe = crop_center(probe, data_dimensions.fov_size)
            plot_probe = probe

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
        else:
            axs4.set_axis_off()
            axs5.set_axis_off()
