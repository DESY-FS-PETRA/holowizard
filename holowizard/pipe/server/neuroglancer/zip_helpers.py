"""A zarr folder zipped for the download."""

import os
import zipfile
from pathlib import Path
from typing import Iterator

from holowizard.pipe.server.neuroglancer.zip_stream import ZipStream


def zip_folder(folder: Path, arc_root: str) -> Iterator[bytes]:
    """The folder as a zip (stored, the chunks are compressed already) under ``arc_root``, piece by piece."""
    sink = ZipStream()
    with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for dirpath, dirnames, filenames in os.walk(folder):
            dirnames.sort()
            for name in sorted(filenames):
                path = Path(dirpath) / name
                zf.write(path, f"{arc_root}/{path.relative_to(folder).as_posix()}")
                yield sink.take()
    yield sink.take()
