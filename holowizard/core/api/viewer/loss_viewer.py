import logging

from holowizard.core.src.visualization.viewer import Viewer


class LossViewer(Viewer):
    def __init__(self):
        super().__init__()

    def update(self, iteration, object, probe, data_dimensions, loss):
        logging.loss(f"{'Reconstruction - '}{'Iter:'}{iteration: < 10}{'Loss:'}{loss[iteration].cpu().numpy(): < 25}")
