import io
import numpy as np
from holowizard.core.reconstruction.viewer.viewer import Viewer
import zmq
import matplotlib
import matplotlib.pyplot as plt
import dotenv
import os

matplotlib.use("Agg")
# 1) Ask dotenv where it *would* look first:
print(os.getcwd())
dotenv_path = dotenv.find_dotenv(usecwd=True)
print("dotenv will load from:", dotenv_path or "<none found>")

# 2) Actually load it (you can also pass verbose=True to get a little feedback)
loaded = dotenv.load_dotenv(dotenv_path, verbose=True, override=True)
print(f"load_dotenv(verbose=True) returned: {loaded}")


# 1) create a single global ZMQ context + PUB socket
zmq_ctx_plotter = zmq.Context.instance()
pub_sock_plotter = zmq_ctx_plotter.socket(zmq.PUB)

pub_sock_plotter.connect(f"tcp://{os.getenv('HNAME')}:{os.getenv('SUB_PORT', '6000')}")
print(f"WebsocketPlotter connected to PUB socket at tcp://{os.getenv('HNAME')}:{os.getenv('SUB_PORT', '6000')}")


class WebsocketPlotter(Viewer):
    def __init__(self, session_id):
        super().__init__()
        self.topic = session_id.encode()  # so subscribers can SUBSCRIBE to only this session
        plt.ioff()
        plt.close("all")
        plt.close()
        plt.pause(0.05)
        self.fig = plt.figure(figsize=(12.8, 8.8))

    
    def draw(self):
        self.fig.clear()
        self.fig.suptitle(self.fig_title)

        self.axs0 = self.fig.add_subplot(122)
        self.axs1 = self.fig.add_subplot(121)

        pos = self.axs0.imshow(self.image, cmap="gray", interpolation="none")
        cbar = self.fig.colorbar(pos, ax=self.axs0, fraction=0.046, pad=0.07)#, orientation="horizontal")
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
        h, w = self.image.shape[:2]

        plt.tight_layout()

        self.fig.canvas.draw()

    def update(self, iteration, x_axis, y_axis, image):
        self.x_axis = np.array(x_axis)
        self.y_axis = y_axis

        self.fig_title = "Find focus with model fit criterion - Running - Iteration " + str(iteration)
        self.axs0_title = "Object at sampling point " + r'$z_{01}$=' + str(round(self.x_axis[-1],1)) + "mm"

        self.image = image

        self.draw()
        self.send_figure()


    def finish(self):
        self.fig_title = "Find focus with model fit criterion - Finished!"
        self.axs0_title = "Object at minimum"

        self.image = self.min_image

        self.draw()
        self.send_figure()


    def send_figure(self):
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        png_bytes = buf.getvalue()
        try:
            pub_sock_plotter.send_multipart([self.topic, png_bytes], flags=zmq.DONTWAIT)
        except zmq.Again:
            pass
