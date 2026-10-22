"""Worker status page (dashboard.html): reconstruction progress of all submitted scans."""

import json

from fastapi import APIRouter, Request
from fastapi.encoders import jsonable_encoder

router = APIRouter(tags=["dashboard"])


def _progress(state):
    scans, queue = state.beamtime.scans, state.beamtime.cluster.queue_info()
    prog = {}
    for scan in reversed(scans):
        key = scan.key
        q = [f for f in queue if key in f["key"]]
        tasks = {}
        for t in scan.config.scan.tasks:
            tq = [f for f in q if t in f["key"]]
            running = sum(1 for f in tq if f["state"] == "processing")
            total = len(scan.hologram_path) if t == "reconstruction" else 1
            if t == "reconstruction":
                # Counted by the beamtime as the images finish (0 before the reconstruction has started).
                done = getattr(scan, "reconstruction_done", 0)
                failed = getattr(scan, "reconstruction_failed", 0)
            else:
                done = int(any(x["name"] == t and x["status"] == "done" for x in scan.done))
                failed = int(any(x["name"] == t and x["status"] == "failed" for x in scan.done))
            tasks[t] = {
                "total": total,
                "done": done,
                "running": running,
                "failed": failed,
            }
        prog[key] = {
            "name": scan.name,
            "base_dir": scan.config.paths.base_dir,
            "path": scan.path_processed,
            "tasks": tasks,
            "cancelled": scan.cancelled,
        }
    return {
        "progress": prog,
        "num_workers": len(state.beamtime.cluster.client_scheduler.nthreads()),
        "cluster": state.cluster.status(),
    }


@router.get("/api/queue")
async def queue_info(request: Request):
    return request.app.state.beamtime.cluster.queue_info()


@router.get("/api/progress")
async def progress(request: Request):
    return _progress(request.app.state)


@router.get("/api/cancel/{scan_id}")
async def cancel_scan(scan_id: str, request: Request):
    request.app.state.beamtime.cancel_task(scan_id)
    return {"status": "cancelled", "scan_id": scan_id}


@router.get("/dashboard")
async def dashboard(request: Request):
    # The page renders the table itself from this data and then updates it from /api/progress (no page reloads).
    # json.dumps keeps the order (newest scan first, tasks in pipeline order); Jinja's tojson would sort the keys.
    # "<", ">" and "&" are escaped so that the JSON is safe inside the <script> element.
    state = request.app.state
    initial = json.dumps(jsonable_encoder(_progress(state)))
    initial = initial.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return state.templates.TemplateResponse("dashboard.html", {"request": request, "initial": initial})
