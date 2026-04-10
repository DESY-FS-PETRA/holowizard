from abc import ABC, abstractmethod


class Plotter(ABC):
    @abstractmethod
    def update(self, iteration, x_axis, y_axis, image): ...

    @abstractmethod
    def finish(self): ...
