import os
import logging


class Paths:
    def __init__(self, root_dir=None, other=None, **kwargs):
        if other is None:
            self.root_dir = root_dir
            self.dirs = {}
            self.files = {}
            self.names = {}
        else:
            if root_dir is None:
                self.root_dir = other.root_dir
            else:
                self.root_dir = root_dir
            self.dirs = other.dirs
            self.files = other.files
            self.names = other.names

        if self.root_dir is None:
            raise RuntimeError("root_dir is empty")

        self.__dict__.update(kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    def mkdirs(self, make_root_dir=False):
        if make_root_dir:
            os.mkdir(self.root_dir)

        for dir in self.dirs:
            logging.info("Creating directory  " + self.dirs[dir])
            try:
                os.makedirs(self.dirs[dir])
            except:
                pass

    @property
    def root_dir(self):
        return self._root_dir

    @root_dir.setter
    def root_dir(self, root_dir):
        self._root_dir = root_dir

    def get_dict(self):
        output = {}
        output["root_dir"] = self.root_dir
        output.update(self.dirs)
        output.update(self.files)
        output.update(self.names)
        return output

    def add_dirs(self, dirs):
        self.dirs.update(dirs)

    def add_files(self, files):
        self.files.update(files)

    def add_names(self, names):
        self.names.update(names)
