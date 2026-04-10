import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from holowizard.core.reconstruction.plotter.plotter import Plotter

class NelderMeadPlotter(Plotter):
    def __init__(self, maximize=True):
        super().__init__()
        self.fig = None
        plt.ion()
        plt.pause(0.05)
        self.fig = plt.figure(figsize=(12.8, 8.8))
        self.fig.tight_layout()

        if maximize:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()

        if matplotlib.rcParams['figure.raise_window']:
            matplotlib.rcParams['figure.raise_window'] = False

    def draw(self):
        self.fig.clear()
        self.fig.suptitle(self.fig_title)

        self.axs0 = self.fig.add_subplot(121)
        self.axs1 = self.fig.add_subplot(122)

        pos = self.axs0.imshow(self.image, cmap="gray", interpolation="none")
        cbar = self.fig.colorbar(pos, ax=self.axs0, fraction=0.046, pad=0.07, orientation="horizontal")
        cbar.ax.set_xlabel(r'$\phi$ / rad')

        self.axs0.set_title(self.axs0_title)
        self.axs0.tick_params(left=False, bottom=False)

        self.axs1.plot(self.x_axis, self.y_axis, marker='.')

        ylims = self.axs1.get_ylim()
        ylims = (ylims[0] - (ylims[1] - ylims[0]) / 6, ylims[1])
        self.axs1.set_ylim(ylims[0],ylims[1])

        for i in range(np.minimum(5,len(self.y_axis))):
            self.axs1.annotate(str(i+1), (self.x_axis[i], self.y_axis[i]), xytext=(0, -20), textcoords='offset points',
                         size='large', horizontalalignment='center')

        min_index = np.argmin(self.y_axis)
        minimum_mm = round(self.x_axis[min_index],1)

        if min_index == len(self.y_axis) - 1:
            self.min_image = self.image

        self.axs1.plot(self.x_axis[min_index],self.y_axis[min_index],'*',color='red')
        self.axs1.set_xlabel(r'$z_{01}$ / mm')
        self.axs1.set_ylabel("MFE / A.U.")

        self.axs1.set_title(r'Sampling, minimum at $z_{01}$=' +str(minimum_mm) + "mm")
        self.fig.canvas.draw()


    def update(self, iteration, x_axis, y_axis, image):
        self.x_axis = np.array(x_axis)/1e6
        self.y_axis = y_axis

        self.fig_title = "Find focus with model fit criterion - Running - Iteration " + str(iteration)
        self.axs0_title = "Object at sampling point " + r'$z_{01}$=' + str(round(self.x_axis[-1],1)) + "mm"

        self.image = image
        self.draw()
        plt.pause(0.01)

    def finish(self):
        self.fig_title = "Find focus with model fit criterion - Finished!"
        self.axs0_title = "Object at minimum"

        self.image = self.min_image

        plt.ioff()
        self.draw()
        plt.show()