"""Password login for the pipe server.

Users are stored with PBKDF2 password hashes in a JSON file (default
``~/.config/holowizard/pipe_users.json``, override with ``HOLOPIPE_USERS_FILE``).
If the file does not exist, it is created on server start with the user
``p05user`` and a random password that is printed once to the console.

Manage users from the command line::

    holowizard_pipe_users list
    holowizard_pipe_users add <user>
    holowizard_pipe_users passwd <user>
    holowizard_pipe_users remove <user>
"""

import argparse
import base64
import getpass
import hashlib
import hmac
import json
import os
import secrets
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import quote

DEFAULT_USER = "p05user"
SESSION_COOKIE = "holopipe_session"
SESSION_MAX_AGE = 7 * 24 * 3600
PBKDF2_ITERATIONS = 600_000
PUBLIC_PATHS = ("/login", "/logout", "/static/")


def users_file() -> Path:
    return Path(os.getenv("HOLOPIPE_USERS_FILE", Path.home() / ".config" / "holowizard" / "pipe_users.json"))


# --- Password hashing ---
def hash_password(password: str, salt: Optional[str] = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def check_password(password: str, stored: str) -> bool:
    try:
        _, iterations, salt, digest = stored.split("$")
    except ValueError:
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(iterations))
    return hmac.compare_digest(candidate.hex(), digest)


# --- User store ---
class UserStore:
    """JSON file with the cookie signing secret and the password hashes, re-read when it changes."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._mtime = None
        self._data = {"secret": "", "users": {}}

    def _load(self) -> Dict:
        with self._lock:
            try:
                mtime = self.path.stat().st_mtime_ns
            except FileNotFoundError:
                return {"secret": "", "users": {}}
            if mtime != self._mtime:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
                self._mtime = mtime
            return self._data

    def _save(self, data: Dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{secrets.token_hex(4)}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)

    def exists(self) -> bool:
        return self.path.exists()

    @property
    def secret(self) -> bytes:
        return self._load()["secret"].encode()

    def users(self) -> Dict[str, str]:
        return self._load()["users"]

    def set_password(self, user: str, password: str):
        data = json.loads(json.dumps(self._load()))
        data["secret"] = data["secret"] or secrets.token_hex(32)
        data["users"][user] = hash_password(password)
        self._save(data)

    def remove(self, user: str):
        data = json.loads(json.dumps(self._load()))
        del data["users"][user]
        self._save(data)

    def verify(self, user: str, password: str) -> bool:
        stored = self.users().get(user)
        if stored is None:
            # Spend the same time as for an existing user.
            check_password(password, hash_password("", "0" * 32))
            return False
        return check_password(password, stored)


def ensure_default_user(store: UserStore):
    """Create the user file with p05user and a random password on first start."""
    if store.exists():
        return
    password = secrets.token_urlsafe(12)
    store.set_password(DEFAULT_USER, password)
    bar = "=" * 72
    print(
        f"\n{bar}\nCreated login user file {store.path}\n"
        f"  user:     {DEFAULT_USER}\n  password: {password}\n"
        f"Change it with: holowizard_pipe_users passwd {DEFAULT_USER}\n{bar}\n",
        flush=True,
    )


# --- Session cookies ---
def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _signature(store: UserStore, payload: str, user: str) -> str:
    # The password hash is part of the key, so changing a password or removing a user ends their sessions.
    key = store.secret + store.users().get(user, "").encode()
    return _b64(hmac.new(key, payload.encode(), hashlib.sha256).digest())


def make_session(store: UserStore, user: str) -> str:
    payload = _b64(json.dumps({"u": user, "exp": int(time.time()) + SESSION_MAX_AGE}).encode())
    return f"{payload}.{_signature(store, payload, user)}"


def session_user(store: UserStore, cookie: Optional[str]) -> Optional[str]:
    if not cookie or "." not in cookie:
        return None
    payload, sig = cookie.rsplit(".", 1)
    try:
        data = json.loads(_unb64(payload))
        user = data["u"]
    except (ValueError, KeyError, TypeError):
        return None
    if user not in store.users() or data.get("exp", 0) < time.time():
        return None
    if not hmac.compare_digest(sig, _signature(store, payload, user)):
        return None
    return user


def _cookies(scope) -> Dict[str, str]:
    from http.cookies import SimpleCookie

    for name, value in scope.get("headers", []):
        if name == b"cookie":
            jar = SimpleCookie()
            try:
                jar.load(value.decode("latin-1"))
            except Exception:
                return {}
            return {k: m.value for k, m in jar.items()}
    return {}


def _header(scope, key: bytes) -> Optional[str]:
    for name, value in scope.get("headers", []):
        if name == key:
            return value.decode("latin-1")
    return None


class LoginRequiredMiddleware:
    """Require a valid session for every HTTP request and websocket except PUBLIC_PATHS."""

    def __init__(self, app, store: UserStore):
        self.app = app
        self.store = store

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        path = scope["path"]
        if scope["type"] == "http" and (path in PUBLIC_PATHS or path.startswith("/static/")):
            return await self.app(scope, receive, send)

        user = session_user(self.store, _cookies(scope).get(SESSION_COOKIE))
        if scope["type"] == "websocket":
            # Reject cross-site websocket connections in addition to the session check.
            origin, host = _header(scope, b"origin"), _header(scope, b"host")
            if user is None or (origin and host and origin.split("://", 1)[-1] != host):
                await receive()
                await send({"type": "websocket.close", "code": 1008})
                return
        elif user is None:
            if path.startswith("/api/"):
                body = b'{"detail":"Not logged in"}'
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode()),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
            target = path + ("?" + scope["query_string"].decode("latin-1") if scope.get("query_string") else "")
            location = f"/login?next={quote(target, safe='')}"
            await send({"type": "http.response.start", "status": 303, "headers": [(b"location", location.encode())]})
            await send({"type": "http.response.body", "body": b""})
            return

        scope.setdefault("state", {})["user"] = user
        return await self.app(scope, receive, send)


# --- Command line user management ---
def _ask_password() -> str:
    while True:
        password = getpass.getpass("New password: ")
        if len(password) < 8:
            print("Use at least 8 characters.")
            continue
        if getpass.getpass("Repeat password: ") != password:
            print("Passwords do not match.")
            continue
        return password


def main(argv=None):
    parser = argparse.ArgumentParser(description=f"Manage login users of the pipe server ({users_file()}).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list users")
    for cmd, text in (("add", "add a user"), ("passwd", "change a user's password"), ("remove", "remove a user")):
        sub.add_parser(cmd, help=text).add_argument("user")
    args = parser.parse_args(argv)

    store = UserStore(users_file())
    users = store.users()
    if args.cmd == "list":
        print("\n".join(sorted(users)) or "no users")
    elif args.cmd == "add":
        if args.user in users:
            sys.exit(f"User {args.user!r} exists, use 'passwd' to change the password.")
        store.set_password(args.user, _ask_password())
        print(f"Added {args.user!r}.")
    elif args.cmd == "passwd":
        if args.user not in users:
            sys.exit(f"No user {args.user!r}.")
        store.set_password(args.user, _ask_password())
        print(f"Password of {args.user!r} changed; existing logins of this user are signed out.")
    elif args.cmd == "remove":
        if args.user not in users:
            sys.exit(f"No user {args.user!r}.")
        if len(users) == 1:
            sys.exit("Cannot remove the last user.")
        store.remove(args.user)
        print(f"Removed {args.user!r}.")


if __name__ == "__main__":
    main()
