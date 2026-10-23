"""Scan detail page (scan.html): tasks and citations of one submitted scan."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from omegaconf import OmegaConf

import holowizard

CITATIONS_FILE = Path(__file__).resolve().parents[2] / "citations.yaml"

# Load citations
citations = OmegaConf.to_container(OmegaConf.load(str(CITATIONS_FILE)), resolve=True).get("citations", {})

router = APIRouter(tags=["scan"])


@router.get("/scan/{name}", response_class=HTMLResponse)
async def scan_detail(name: str, request: Request):
    state = request.app.state
    scan = state.beamtime.get_scan(name)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    return state.templates.TemplateResponse(
        "scan.html",
        {
            "request": request,
            "scan": scan,
            "citations": citations,
            "hw": {"version": holowizard.__version__},
        },
    )
