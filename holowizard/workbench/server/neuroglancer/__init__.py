"""Tomograms as OME-Zarr for the Neuroglancer page (routes in ``server/routes/neuroglancer.py``).

Tomograms (the TIFF slices in ``path_processed/<scan>/recon_dir/<version>``, as in the scan overview) are
converted to a multiscale OME-Zarr (zarr v2, chunks of 64³ voxels, blosc-zstd; every level is downsampled by 2
until the largest side is at most 128 voxels). Everything is stored in the cache folder
``<beamtime log_path>/neuroglancer`` (``scratch_cc/holowizard_workbench/neuroglancer``):

* ``<id>.zarr``: the data, served to the viewer under ``/neuroglancer/data/<id>/``
* ``<id>.json``: what it is (source tomogram, user, time, value range for the display)

Optionally, voxels outside the reconstruction cylinder (the circle inscribed in the slices) are masked: they get
a "no data" value (0 in 8 bit zarrs, whose data then use 1..255; the most negative float32 in 32 bit zarrs),
which is also the zarr fill value, so chunks outside the cylinder are not stored. The viewer's shader draws
them transparent and they are left out of the downsampling and the value statistics.

The Neuroglancer client is the prebuilt one shipped with the ``neuroglancer`` Python package (only its static
files are used, the package is not imported), so the viewer works without internet access. That build is the
client of Neuroglancer's Python integration: it is loaded from ``/neuroglancer/v/<token>/`` and talks to a
Python server next to it (a long-lived ``events`` stream and ``state``/``action`` posts). Those are answered
inside the browser by a small script added to its index.html, so no connection stays open (browsers allow only
about six per server, shared with all other pages). The page sets the viewer state itself (URL hash and
``window.viewer`` of the same-origin iframe).

Modules:

* ``constants``: settings of the conversion, the cache and the value statistics
* ``zarr_job``: ``ZarrJob``, the conversion of one tomogram (in a thread)
* ``job_registry``: ``JobRegistry``, the conversions of this server process
* ``convert_request``: ``ConvertRequest``, the body of a conversion request
* ``zip_stream``: ``ZipStream``, the sink for streaming a zip
* ``zarr_helpers``: OME-Zarr metadata, reading zarr v2 arrays and value statistics
* ``array_helpers``: downsampling and the cylinder mask
* ``cache_helpers``: the entries of the cache folder (ids, meta data, listing)
* ``zip_helpers``: zipping a zarr for the download
* ``client_helpers``: the Neuroglancer client (its folder, the in-browser server shim, the scale bar patch)
* ``request_helpers``: the user of a request
"""
