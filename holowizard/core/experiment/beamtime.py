import os

from holowizard.core.utils.fileio import load_scan_params
from holowizard.core.experiment.scan import P05_Scan


class Beamtime:
    def __init__(self, id, year, beamline="p05", bt_type="data"):
        self.bt_dict = {}
        self.id = str(id)
        self.year = str(year)
        self.beamline = str(beamline)
        self.bt_type = str(bt_type)
        self.path_raw = (
            "/asap3/petra3/gpfs/"
            + self.beamline
            + "/"
            + self.year
            + "/"
            + self.bt_type
            + "/"
            + self.id
            + "/raw/"
        )
        self.path_processed = (
            "/asap3/petra3/gpfs/"
            + self.beamline
            + "/"
            + self.year
            + "/"
            + self.bt_type
            + "/"
            + self.id
            + "/processed/"
        )
        self.energy = 17
        self.holder = 190
        self.scans = []
        self.scan_list = self.read_scanlist()

    def read_scanlist(self):
        scanlist = os.listdir(self.path_raw)
        scanlist.sort()
        scans_tab = {}
        for i, scan in enumerate(scanlist):
            if "nano" in scanlist[i]:
                try:
                    scans_tab[f"{scan}"] = load_scan_params(self.path_raw + "/" + scan)
                except:
                    scans_tab[f"{scan}"] = "NA"
        return scans_tab

    def create_scans(self):
        for scan in self.scan_list:
            self.add_scan(scan)

    def remove_scan(self):
        return

    def add_scan(self, scan, energy=17, holder=195):
        self.scans.append(P05_Scan(scan, energy, holder, self.path_raw))

    def set_energy_global(self, energy):
        self.energy = energy

    #    @property
    #    def scan_list(self):
    #        return self.scan_list

    @property
    def path_raw(self):
        return self._path_raw

    @property
    def path_processed(self):
        return self._path_processed

    @path_raw.setter
    def path_raw(self, path_new):
        self._path_raw = path_new

    @path_processed.setter
    def path_processed(self, path_new):
        self._path_processed = path_new
