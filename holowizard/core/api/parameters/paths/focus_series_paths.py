from .paths import Paths


class FocusSeriesPaths(Paths):
    def __init__(self, root_dir=None, other: Paths = None, **kwargs):
        super().__init__(root_dir, other)

        if other is not None:
            self.__dict__.update(other.get_dict())

        self.dirs["focus_series_dir"] = self.root_dir
        self.dirs["projections_dir"] = self.root_dir + "/projections/"
        self.dirs["se_losses"] = self.root_dir + "/se_losses/"

        self.__dict__.update(kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    @property
    def focus_series_dir(self):
        return self.dirs["focus_series_dir"]

    @property
    def projections_dir(self):
        return self.dirs["projections_dir"]

    @property
    def se_losses(self):
        return self.dirs["se_losses"]
