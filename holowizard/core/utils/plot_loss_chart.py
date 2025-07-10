import glob
import torch
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import holowizard.core
from holowizard.core.find_focus import focus_loss_metrics
import os
from scipy.ndimage import fourier_gaussian


def print_loss_csv_normalized(file_name, x_values, loss_values):
    y_min = np.min(loss_values)
    y_max = np.max(loss_values) - y_min

    f = open(file_name, "w")
    for i in range(len(x_values)):
        f.write(str(x_values[i]) + "," + str((loss_values[i] - y_min) / y_max) + "\n")


def print_loss_csv(file_name, x_values, loss_values):
    f = open(file_name, "w")
    for i in range(len(x_values)):
        f.write(str(x_values[i]) + "," + str(loss_values[i]) + "\n")


def plot_loss_chart(root, support, title, csv_output=False):
    fig, axs = plt.subplots(2, 4)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    fig.suptitle(title)

    se_loss_files = glob.glob(root + "/se_losses/" + "loss_*")
    image_files = glob.glob(root + "/projections/" + "img_*")

    se_loss_files.sort()
    image_files.sort()

    x_values_se = []
    se_values = []

    for file in se_loss_files:
        with open(file, "r") as f:
            line = f.read()
            line = line.split("(")
            line = line[1].split(",")
            x = line[0]
            y = line[1].split(")")[0]
            x_values_se.append(float(x))
            se_values.append(float(y))

    x_values_images = []
    gog_values = []
    tog_values = []
    var_values = []
    spec_values = []
    gc_values = []
    lp_values = []

    for file in image_files:
        x = file.split("/")[-1]
        x = x.split("_")[1]
        x = x.split(".")[0]
        x = int(x)

        x_values_images.append(x)

        image = torch.tensor(
            io.imread(file), device=holowizard.core.torch_running_device
        )[support]

        """
		sample_grid = torch.meshgrid(torch.fft.fftfreq(image.shape[0], device=torch.device("cuda")),
									 torch.fft.fftfreq(image.shape[1], device=torch.device("cuda")))
		xi, eta = sample_grid
		# Butterworth bandpass
		denom1 = xi * xi + eta * eta - 0.35 ** 2
		denom2 = 1 + (0.09 / denom1) ** 10
		bandpass = 1 - 1 / denom2
		image = torch.fft.ifft2(torch.fft.fft2(image) * bandpass).real
		"""

        filter_kernel = torch.tensor(
            fourier_gaussian(np.ones(image.shape), sigma=4 / 2.35)[
                :, 0 : int(image.shape[1] / 2) + 1
            ],
            device=holowizard.core.torch_running_device,
        )

        values_real_fft = torch.fft.rfft2(image)
        values_real_fft *= filter_kernel
        image = torch.fft.irfft2(values_real_fft, image.size())

        gog_values.append(focus_loss_metrics.get_gog(image))
        tog_values.append(focus_loss_metrics.get_tog(image))
        var_values.append(focus_loss_metrics.get_var(image))
        spec_values.append(focus_loss_metrics.get_spec(image))
        gc_values.append(focus_loss_metrics.get_gra(image))
        lp_values.append(focus_loss_metrics.get_lap(image))

        if x == round(len(image_files) / 2):
            axs[0, 0].imshow(image.cpu().numpy(), cmap="gray", interpolation="none")
            axs[0, 0].set_title("Input at distance " + "{:.2e}".format(x_values_se[x]))

    x_values_se, se_values = zip(*sorted(zip(x_values_se, se_values)))
    x_values_gog, gog_values = zip(*sorted(zip(x_values_images, gog_values)))
    x_values_tog, tog_values = zip(*sorted(zip(x_values_images, tog_values)))
    x_values_var, var_values = zip(*sorted(zip(x_values_images, var_values)))
    x_values_spec, spec_values = zip(*sorted(zip(x_values_images, spec_values)))
    x_values_lp, lp_values = zip(*sorted(zip(x_values_images, lp_values)))
    x_values_gc, gc_values = zip(*sorted(zip(x_values_images, gc_values)))

    print(title)

    index_min = min(range(len(se_values)), key=se_values.__getitem__)

    print(
        "Minimum: Index=",
        index_min,
        "    SE=",
        se_values[index_min],
        "    z01=",
        x_values_se[index_min],
    )

    if csv_output == True:
        csv_root = root + "/csv/"
        try:
            os.mkdir(csv_root)
        except:
            None

        print_loss_csv(csv_root + "se_losses.csv", x_values_se, se_values)
        print_loss_csv_normalized(
            csv_root + "se_losses_normalised.csv", x_values_se, se_values
        )

        print_loss_csv(csv_root + "gog_losses.csv", x_values_se, gog_values)
        print_loss_csv_normalized(
            csv_root + "gog_losses_normalised.csv", x_values_se, gog_values
        )

        print_loss_csv(csv_root + "tog_losses.csv", x_values_se, tog_values)
        print_loss_csv_normalized(
            csv_root + "tog_losses_normalised.csv", x_values_se, tog_values
        )

        print_loss_csv(csv_root + "var_losses.csv", x_values_se, var_values)
        print_loss_csv_normalized(
            csv_root + "var_losses_normalised.csv", x_values_se, var_values
        )

        print_loss_csv(csv_root + "spec_losses.csv", x_values_se, spec_values)
        print_loss_csv_normalized(
            csv_root + "spec_losses_normalised.csv", x_values_se, spec_values
        )

        print_loss_csv(csv_root + "lp_losses.csv", x_values_se, lp_values)
        print_loss_csv_normalized(
            csv_root + "lp_losses_normalised.csv", x_values_se, lp_values
        )

        print_loss_csv(csv_root + "gc_losses.csv", x_values_se, gc_values)
        print_loss_csv_normalized(
            csv_root + "gc_losses_normalised.csv", x_values_se, gc_values
        )

    axs[0, 1].plot(x_values_se, se_values)
    axs[0, 1].set_title("SE")

    axs[0, 2].plot(x_values_se, gog_values)
    axs[0, 2].set_title("GOG")

    axs[0, 3].plot(x_values_se, tog_values)
    axs[0, 3].set_title("TOG")

    axs[1, 0].plot(x_values_se, var_values)
    axs[1, 0].set_title("VAR")

    axs[1, 1].plot(x_values_se, spec_values)
    axs[1, 1].set_title("SPEC")

    axs[1, 2].plot(x_values_se, gc_values)
    axs[1, 2].set_title("GRA")

    axs[1, 3].plot(x_values_se, lp_values)
    axs[1, 3].set_title("LAP")


def plot_loss_chart_multidim(
    root, support, title, csv_output=False, xmin=None, xmax=None, ymin=None, ymax=None
):
    axs = []
    fig = plt.figure()
    axs.append(fig.add_subplot(1, 1, 1, projection="3d"))
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    fig.suptitle(title)

    se_loss_files = glob.glob(root + "/se_losses/" + "loss_*")
    se_loss_files.sort()

    x_values_se = []
    y_values_se = []
    se_values = []

    for file in se_loss_files:
        with open(file, "r") as f:
            line = f.read()
            line = line.split("(")
            line = line[1].split(",")
            x = line[0]
            y = line[1]
            z = line[2].split(")")[0]

            if xmax is not None and float(x) > xmax:
                continue
            if xmin is not None and float(x) < xmin:
                continue
            if ymax is not None and float(y) > ymax:
                continue
            if ymin is not None and float(y) < ymin:
                continue
            x_values_se.append(float(x))
            y_values_se.append(float(y))
            se_values.append(float(z))

    x_values_se, y_values_se, se_values = zip(
        *sorted(zip(x_values_se, y_values_se, se_values))
    )

    index_min = min(range(len(se_values)), key=se_values.__getitem__)

    print(
        "Minimum: Index=",
        index_min,
        "    SE=",
        se_values[index_min],
        "    z01=",
        x_values_se[index_min],
        "    a0=",
        y_values_se[index_min],
    )

    axs[0].plot(x_values_se, y_values_se, se_values, "*")
    axs[0].set_title("SE")

    axs[0].set_xlabel("z01")
    axs[0].set_ylabel("a0")
