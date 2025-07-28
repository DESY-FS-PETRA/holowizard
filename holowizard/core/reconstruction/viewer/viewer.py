from abc import ABC, abstractmethod


class Viewer(ABC):
    @abstractmethod
    def update(self, iteration, data, probe, data_dimensions, loss): ...
