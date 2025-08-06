import io
from holowizard.core.reconstruction.viewer.viewer import Viewer
import zmq
import matplotlib
import dotenv
import os
import matplotlib.pyplot as plt

from holowizard.core.utils.transform import crop_center

matplotlib.use("Agg")

# 1) Ask dotenv where it *would* look first:
dotenv_path = dotenv.find_dotenv()
print("dotenv will load from:", dotenv_path or "<none found>")

# 2) Actually load it (you can also pass verbose=True to get a little feedback)
loaded = dotenv.load_dotenv(dotenv_path, verbose=True, override=True)
print(f"load_dotenv(verbose=True) returned: {loaded}")


# 1) create a single global ZMQ context + PUB socket
zmq_ctx = zmq.Context.instance()
pub_sock = zmq_ctx.socket(zmq.PUB)

pub_sock.connect(f"tcp://{os.getenv('HNAME')}:{os.getenv('SUB_PORT', '6000')}")
print(f"WebsocketViewer connected to PUB socket at tcp://{os.getenv('HNAME')}:{os.getenv('SUB_PORT', '6000')}")


class WebsocketViewer(Viewer):
    def __init__(self, session_id):
        super().__init__()
        self.topic = session_id.encode()  # so subscribers can SUBSCRIBE to only this session
        plt.ioff()
        plt.close("all")
        plt.close()
        plt.pause(0.05)
        self.fig = plt.figure(figsize=(12.8, 8.8))

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
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        png_bytes = buf.getvalue()
        try:
            pub_sock.send_multipart([self.topic, png_bytes], flags=zmq.DONTWAIT)
        except zmq.Again:
            pass
