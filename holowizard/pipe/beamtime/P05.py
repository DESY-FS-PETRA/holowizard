import os
import numpy as np
from holowizard.pipe.beamtime import Beamtime

class P05Beamtime(Beamtime):
    """
    Concrete scan object for P05 beamtime, inherits BeamtimeObject.
    """
    def __init__(self, beamtime_name, year, cluster):
        """
        Initialize the P05Beamtime object.

        Args:
            path_raw (str): Path to raw data.
            path_processed (str): Path to processed data.
            beamtime_name (str): Name of the beamtime, default is 'P05'.
        """

        path_raw = f"/asap3/petra3/gpfs/p05/{year}/data/{beamtime_name}/raw"
        path_processed= f"/asap3/petra3/gpfs/p05/{year}/data/{beamtime_name}/processed"
        log_path = f"/asap3/petra3/gpfs/p05/{year}/data/{beamtime_name}/scratch_cc"
        super().__init__(path_raw, path_processed, log_path, beamtime_name, cluster)
        #self.meta_dict = self._load_metadata(path_raw)
        
    def _load_metadata(self, path) -> dict:
        logfile = os.path.join(path, f'{self.beamtime_name}__ScanParam.txt')
        metadata = {}
        try:
            with open(logfile) as file:
                for line in file:
                    if ': ' in line:
                        key, value = line.strip().split(': ', 1)
                        metadata[key] = value
        except FileNotFoundError:
            metadata[self.name] = "Metadata file not found."
        except Exception as e:
            metadata[self.name] = f"Metadata read error: {e}"
        return metadata

  