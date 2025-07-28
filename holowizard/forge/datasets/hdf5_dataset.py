# standard libraries
from typing import List

# third party libraries
from torch.utils.data import Dataset
import h5py

# local libraries
from holowizard.forge.utils.datatypes import Pathlike, TensorFloat32


__all__ = [
    "HDF5Dataset",
]


class HDF5Dataset(Dataset):
    """Dataset for an HDF5 file."""

    def __init__(
        self,
        filename: Pathlike,
        hdf5_dataset_name: str,
        hdf5_group_names: List[str] = [],
        transform=None,
    ):
        """
        Args:
            filename (Pathlike): Path to the HDF5-file containing the realworld flat fields.
            hdf5_dataset_name (str): Name of the dataset in the hdf5-file.
            transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
        """
        super(HDF5Dataset, self).__init__()
        self.filename = filename
        self.hdf5_group_names = hdf5_group_names
        self.hdf5_dataset_name = hdf5_dataset_name
        self.transform = transform

        self.hdf5_file = h5py.File(self.filename, "r")
        if self.hdf5_group_names != []:
            sub_group = self.hdf5_file
            for group_name in self.hdf5_group_names:
                sub_group = sub_group[group_name]
            self.dataset = sub_group[self.hdf5_dataset_name]
        else:
            self.dataset = self.hdf5_file[self.hdf5_dataset_name]
        self.length = len(self.dataset)

    def __len__(self) -> int:  # TODO: Store Flatfield
        return self.length

    def __getitem__(self, idx: int) -> TensorFloat32:
        sample = self.dataset[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
