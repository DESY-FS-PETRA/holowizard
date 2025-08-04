import matplotlib
import matplotlib.pyplot as plt

from holowizard.core.utils.plot_loss_chart import plot_loss_chart

matplotlib.use("Qt5Agg")

# datasets = ["cactus_needle","magnesium_wire","magnesium_wire_insitu_cell","spider_hair", "tooth"]
datasets = ["cactus_needle"]

for dataset in datasets:
    root = "/gpfs/petra3/scratch/dorajoha/focus_series_singledim/" + dataset
    support = [slice(0, 2048), slice(0, 2048)]
    plot_loss_chart(root, support, dataset, True)
    plt.show()
