"""Body of a conversion request (tomogram -> OME-Zarr)."""

from pydantic import BaseModel

from holowizard.workbench.server.neuroglancer.constants import DEFAULT_MASK_RADIUS


class ConvertRequest(BaseModel):
    dataset: str
    dtype: str = "uint8"
    mask: bool = True  # mask the voxels outside the reconstruction cylinder
    mask_radius: float = DEFAULT_MASK_RADIUS  # fraction of the inscribed circle
