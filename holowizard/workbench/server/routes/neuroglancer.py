"""Neuroglancer page (neuroglancer.html): tomograms as OME-Zarr in the Neuroglancer viewer.

Only the routes; the conversion, the cache folder, the zarr helpers and the Neuroglancer client are in the package
``holowizard.workbench.server.neuroglancer`` (see its docstring).
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

from holowizard.workbench.server.neuroglancer.cache_helpers import (
    entry_file,
    entry_path,
    entry_state,
    list_entries,
    read_meta,
    safe_name,
    tomogram_cache_id,
)
from holowizard.workbench.server.neuroglancer.client_helpers import (
    CLIENT_TOKEN,
    client_dir,
    client_index_html,
    is_main_script,
    patched_main_script,
)
from holowizard.workbench.server.neuroglancer.constants import DTYPES
from holowizard.workbench.server.neuroglancer.convert_request import ConvertRequest
from holowizard.workbench.server.neuroglancer.job_registry import registry
from holowizard.workbench.server.neuroglancer.request_helpers import user_name
from holowizard.workbench.server.neuroglancer.zarr_helpers import cached_value_stats
from holowizard.workbench.server.neuroglancer.zarr_job import ZarrJob
from holowizard.workbench.server.neuroglancer.zip_helpers import zip_folder
from holowizard.workbench.server.routes.overview import SOURCES, _cached_by_mtime, _list_tiffs, add_pixel_sizes

router = APIRouter(tags=["neuroglancer"])


# --- Data sets and conversion ---
@router.get("/api/neuroglancer/datasets")
async def neuroglancer_datasets(request: Request):
    return await run_in_threadpool(list_entries, request.app.state.beamtime)


@router.post("/api/neuroglancer/convert")
async def convert(req: ConvertRequest, request: Request):
    beamtime, user = request.app.state.beamtime, user_name(request)
    if not 0.5 <= req.mask_radius <= 1.0:
        raise HTTPException(status_code=400, detail="The mask radius must lie between 50 and 100 %")
    if req.dtype not in DTYPES:
        raise HTTPException(status_code=400, detail=f"dtype must be one of {', '.join(DTYPES)}")
    src = SOURCES["tomo"]
    if not src.valid_id(req.dataset):
        raise HTTPException(status_code=400, detail=f"Invalid tomogram: {req.dataset!r}")
    folder = src.folder(src.root(beamtime), req.dataset)
    files = await run_in_threadpool(_cached_by_mtime, folder, _list_tiffs)
    if not files:
        raise HTTPException(status_code=404, detail=f"No slices found in {folder}")
    cache_id = tomogram_cache_id(req.dataset)
    running = registry.running_conflict(cache_id, user)
    if running is not None:
        what = "This tomogram is" if running.cache_id == cache_id else "You are"
        raise HTTPException(status_code=409, detail=f"{what} already being converted ({running.dataset}).")
    summary = {"name": req.dataset, "count": len(files)}
    await run_in_threadpool(add_pixel_sizes, beamtime, "tomo", [summary])
    job = ZarrJob(
        beamtime,
        user,
        req.dataset,
        folder,
        files,
        req.dtype,
        summary.get("pixel_size_nm"),
        summary.get("pixel_note", ""),
        req.mask,
        req.mask_radius,
    )
    registry.add(job)
    return job.state()


@router.get("/api/neuroglancer/job/{job_id}")
async def job_state(job_id: str):
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Conversion not found (server restarted?)")
    return job.state()


@router.post("/api/neuroglancer/job/{job_id}/cancel")
async def cancel_job(job_id: str, request: Request):
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Conversion not found")
    if job.user != user_name(request):
        raise HTTPException(status_code=403, detail="Only the user who started a conversion can cancel it")
    job.cancelled = True
    return job.state()


# --- Download ---
@router.get("/api/neuroglancer/download/{cache_id}")
async def download(cache_id: str, request: Request):
    folder = entry_path(request.app.state.beamtime, cache_id)
    meta = read_meta(folder.parent, cache_id)
    if meta is None or not folder.is_dir():
        raise HTTPException(status_code=404, detail="Not found")
    if entry_state(meta).get("status") != "done":
        raise HTTPException(status_code=409, detail="The conversion is not finished")
    arc_root = f"{safe_name(meta.get('name', cache_id).replace('/', '_'))}.zarr"
    return StreamingResponse(
        zip_folder(folder, arc_root),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{arc_root}.zip"'},
    )


# --- Contrast ranges (Min-Max, 1-99 %, 5-95 %) over the whole volume ---
@router.get("/api/neuroglancer/stats/{cache_id}")
async def value_stats(cache_id: str, request: Request):
    folder = entry_path(request.app.state.beamtime, cache_id)
    meta_file = folder.parent / f"{cache_id}.json"
    try:
        meta_mtime_ns = meta_file.stat().st_mtime_ns  # changes when the zarr is converted again
    except OSError:
        raise HTTPException(status_code=404, detail="Not found")
    try:
        return await run_in_threadpool(cached_value_stats, str(folder), meta_mtime_ns)
    except (OSError, ValueError, KeyError, IndexError) as e:
        raise HTTPException(status_code=422, detail=f"Could not compute the value range: {e}")


# --- Zarr files for the viewer ---
@router.api_route("/neuroglancer/data/{cache_id}/{path:path}", methods=["GET", "HEAD"])
def zarr_file(cache_id: str, path: str, request: Request):
    """Files of a cached zarr, for the viewer; missing chunks are 404 (the viewer shows the fill value)."""
    target = entry_file(entry_path(request.app.state.beamtime, cache_id), path)
    if target is None:
        raise HTTPException(status_code=404, detail="Not found")
    # Revalidated on every use (ETag), since a tomogram can be converted again under the same id.
    return FileResponse(target, headers={"Cache-Control": "no-cache"})


# --- Neuroglancer client and the server API of its Python integration build ---
@router.get("/neuroglancer/v/{token}/{file:path}")
def client_file(token: str, file: str):
    folder = client_dir()
    if folder is None or not CLIENT_TOKEN.fullmatch(token):
        raise HTTPException(status_code=404, detail="Not found")
    name = file or "index.html"
    path = folder / name
    if name != path.name or not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    if name == "index.html":
        return HTMLResponse(client_index_html(path), headers={"Cache-Control": "no-cache"})
    if is_main_script(name):
        script = patched_main_script(str(path), path.stat().st_mtime_ns)
        return Response(script, media_type="text/javascript", headers={"Cache-Control": "max-age=86400"})
    media_type = "application/wasm" if name.endswith(".wasm") else None
    # The scripts have content hashes in their names, so they can be cached.
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "max-age=86400"})


@router.post("/neuroglancer/state/{token}")
async def client_state(request: Request):
    """Fallback for state changes that reach the server (normally answered in the browser); not stored."""
    try:
        generation = (await request.json()).get("g")
    except ValueError:
        generation = None
    return {"g": str(generation)}


@router.post("/neuroglancer/action/{token}")
async def client_action():
    return {}


# --- Page ---
@router.get("/neuroglancer")
async def neuroglancer_page(request: Request):
    return request.app.state.templates.TemplateResponse(
        "neuroglancer.html", {"request": request, "client_available": client_dir() is not None}
    )
