from abc import ABC, abstractmethod


class Viewer(ABC):
    @abstractmethod
    def update(self, iteration, hologram_measured, hologram_guess, object, probe, data_dimensions, loss): ...
