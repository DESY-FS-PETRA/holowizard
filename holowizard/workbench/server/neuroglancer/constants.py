"""Settings of the OME-Zarr conversion, the cache folder and the value statistics."""

import re

import numpy as np

CACHE_SUBDIR = "neuroglancer"  # in beamtime.log_path
CHUNK = 64  # voxels per chunk side
TOP_LEVEL_SIZE = 128  # downsample until the largest side is at most this
COMPRESSOR = {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 1, "blocksize": 0}
DTYPES = ("uint8", "float32")
# Display range of the converted data: percentiles of a sample of slices. 8 bit copies map a wider range to
# 0..255, so that the contrast can still be changed in the viewer.
DISPLAY_PERCENTILES = (1, 99)
UINT8_PERCENTILES = (0.1, 99.9)
SAMPLE_SLICES = 24
READ_THREADS = 32
WRITE_THREADS = 16
MAX_RANGE_VOXELS = 64 * 1024**2  # largest zarr level read in one piece (value statistics)
NO_DATA_FLOAT32 = float(np.finfo(np.float32).min)  # masked voxels in 32 bit zarrs
NO_DATA_UINT8 = 0  # masked voxels in 8 bit zarrs
DEFAULT_MASK_RADIUS = 0.97  # of the inscribed circle; FBP leaves a bright rim at the edge of the cylinder
STATS_MAX_VOXELS = 256**3  # contrast buttons: statistics of the finest level up to this size
STATS_PERCENTILES = {"minmax": (0, 100), "1-99": (1, 99), "5-95": (5, 95)}
ID_PATTERN = re.compile(r"[A-Za-z0-9_.+-]{1,200}")
