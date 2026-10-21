"""Helpers for the requests of the Neuroglancer routes."""

from fastapi import Request


def user_name(request: Request) -> str:
    """The logged-in user (set by the login middleware)."""
    return getattr(request.state, "user", None) or "anonymous"
