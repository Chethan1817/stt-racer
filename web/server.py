"""Serve the web UI and hand out tokens for the local LiveKit server.

livekit-server --dev      # terminal 1
uv run agent.py dev       # terminal 2
uv run web/server.py      # terminal 3, then open http://localhost:8080
"""

from __future__ import annotations

import os
import secrets
import warnings
from pathlib import Path

from aiohttp import web
from dotenv import load_dotenv
from livekit import api

load_dotenv()
# livekit-server --dev uses the 6-byte secret "secret"; fine for local use.
warnings.filterwarnings("ignore", message="The HMAC key is")

HERE = Path(__file__).parent
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")


async def token(request: web.Request) -> web.Response:
    """A fresh room per visit (or ?room=name); the agent is dispatched to it automatically."""
    room = request.query.get("room") or f"fusion-{secrets.token_hex(3)}"
    identity = f"user-{secrets.token_hex(3)}"
    jwt = (
        api.AccessToken(API_KEY, API_SECRET)
        .with_identity(identity)
        .with_name("You")
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )
    return web.json_response({"url": LIVEKIT_URL, "token": jwt, "room": room})


def static(name: str) -> web.RouteDef:
    async def handler(_: web.Request) -> web.FileResponse:
        return web.FileResponse(HERE / name, headers={"Cache-Control": "no-cache"})

    return web.get("/" if name == "index.html" else f"/{name}", handler)


def main() -> None:
    app = web.Application()
    app.add_routes([static("index.html"), static("app.js"), static("style.css")])
    app.add_routes([web.get("/token", token)])
    port = int(os.getenv("WEB_PORT", "8080"))
    print(f"STT Fusion UI: http://localhost:{port}  (LiveKit server: {LIVEKIT_URL})", flush=True)
    web.run_app(app, host="localhost", port=port, print=None)


if __name__ == "__main__":
    main()
