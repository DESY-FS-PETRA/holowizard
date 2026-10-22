"""Login page (login.html): password login and logout."""

import asyncio

from fastapi import APIRouter, Form, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse

from holowizard.workbench.server.auth import DEFAULT_USER, SESSION_COOKIE, SESSION_MAX_AGE, make_session

router = APIRouter(tags=["login"])


def _safe_next(target: str) -> str:
    # Only redirect to paths on this server.
    return target if target.startswith("/") and not target.startswith("//") else "/"


@router.get("/login")
async def login_page(request: Request, next: str = "/"):
    return request.app.state.templates.TemplateResponse(
        "login.html", {"request": request, "user": DEFAULT_USER, "next": _safe_next(next), "error": None}
    )


@router.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...), next: str = Form("/")):
    store = request.app.state.users
    if not await run_in_threadpool(store.verify, username, password):
        await asyncio.sleep(1)  # slow down password guessing
        return request.app.state.templates.TemplateResponse(
            "login.html",
            {"request": request, "user": username, "next": _safe_next(next), "error": "Wrong user name or password."},
            status_code=401,
        )
    response = RedirectResponse(url=_safe_next(next), status_code=303)
    response.set_cookie(
        SESSION_COOKIE,
        make_session(store, username),
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    return response


@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE)
    return response
