import matplotlib
import matplotlib.pyplot as plt

from holowizard.core.utils.plot_loss_chart import *

matplotlib.use("Qt5Agg")

data_set = "cactus_needle_good_fov_only_2x"

# z01: 86708500.0 a0: 0.928125

root = "/gpfs/petra3/scratch/dorajoha/focus_series_multidim/" + data_set

support = [slice(0, 2048), slice(0, 2048)]

plot_loss_chart_multidim(root, support, data_set, False, ymin=0.3, ymax=0.99)
# plot_loss_chart(root,support,"")
plt.show()
