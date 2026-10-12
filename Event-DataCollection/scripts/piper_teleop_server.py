"""Browser input and preview transport; never touches Isaac Sim objects.

The simulation thread owns the environment. This module shares only small
immutable commands and JPEG bytes with an aiohttp thread.
"""
from __future__ import annotations

import asyncio
import base64
from collections import deque
import json
from pathlib import Path
import secrets
import threading
import time

from aiohttp import WSMsgType, web


KEYS = frozenset("wsadqezxtgcv")
COMMANDS = frozenset(("pause", "resume", "reset", "discard", "save", "grip"))
MAX_JOG_STEPS = 8


class TeleopState:
    def __init__(self, *, pos_step: float, rot_step: float, timeout_s: float = 0.35):
        if not 0 < pos_step <= 0.05 or not 0 < rot_step <= 0.1 or not 0 < timeout_s <= 1.0:
            raise ValueError("teleop requires pos_step<=0.05, rot_step<=0.1 and timeout_s<=1")
        self.pos_step, self.rot_step, self.timeout_s = pos_step, rot_step, timeout_s
        self._lock = threading.Lock()
        self._connected = False
        self._paused = True
        self._keys: frozenset[str] = frozenset()
        self._gripper = 1.0
        self._last_input = 0.0
        self._commands: deque[str] = deque()
        self._jog_key: str | None = None
        self._jog_remaining = 0

    def connect(self) -> bool:
        with self._lock:
            if self._connected:
                return False
            self._connected, self._paused = True, True
            self._keys = frozenset()
            self._jog_key, self._jog_remaining = None, 0
            self._last_input = time.monotonic()
            return True

    def disconnect(self) -> None:
        with self._lock:
            self._connected, self._paused = False, True
            self._keys = frozenset()
            self._jog_key, self._jog_remaining = None, 0
            self._commands.clear()

    def input(self, keys: object) -> None:
        if not isinstance(keys, list) or len(keys) > len(KEYS) or any(
            not isinstance(key, str) or key not in KEYS for key in keys
        ):
            raise ValueError("invalid keyboard input")
        with self._lock:
            self._keys = frozenset(keys)
            self._last_input = time.monotonic()

    def command(self, name: str) -> None:
        if name not in COMMANDS:
            raise ValueError("invalid teleop command")
        with self._lock:
            self._last_input = time.monotonic()
            if name == "grip":
                self._gripper *= -1.0
            elif name == "pause":
                self._paused = True
                self._keys = frozenset()
                self._jog_key, self._jog_remaining = None, 0
            elif name == "resume":
                self._paused = False
                self._jog_key, self._jog_remaining = None, 0
            else:
                self._commands.append(name)
                if name in ("reset", "discard", "save"):
                    self._paused = True
                    self._keys = frozenset()
                    self._jog_key, self._jog_remaining = None, 0

    def jog(self, key: str, steps: int) -> None:
        if key not in KEYS or type(steps) is not int or not 1 <= steps <= MAX_JOG_STEPS:
            raise ValueError("invalid jog; key must be a control key and steps must be 1..8")
        with self._lock:
            if not self._connected:
                raise ValueError("no browser operator connected")
            self._keys = frozenset()
            self._jog_key, self._jog_remaining = key, steps
            self._paused = False
            self._last_input = time.monotonic()

    def take_commands(self) -> list[str]:
        with self._lock:
            commands = list(self._commands)
            self._commands.clear()
            return commands

    def action(self) -> tuple[float, ...] | None:
        """None means freeze the simulation, not a recorded zero-action step."""
        with self._lock:
            if not self._connected or self._paused:
                return None
            if self._jog_remaining:
                keys = frozenset((self._jog_key,))
                self._jog_remaining -= 1
                if self._jog_remaining == 0:
                    self._jog_key = None
                    self._paused = True
            elif time.monotonic() - self._last_input > self.timeout_s:
                self._paused = True
                self._keys = frozenset()
                return None
            else:
                keys = self._keys
            grip = self._gripper
        def axis(positive: str, negative: str) -> float:
            return float(positive in keys) - float(negative in keys)
        position = [axis("w", "s"), axis("d", "a"), axis("e", "q")]
        rotation = [axis("x", "z"), axis("t", "g"), axis("v", "c")]
        # Normalize diagonals so holding two keys cannot increase command norm.
        for vector in (position, rotation):
            norm = sum(value * value for value in vector) ** 0.5
            if norm > 1:
                vector[:] = [value / norm for value in vector]
        return tuple(value * self.pos_step for value in position) + tuple(
            value * self.rot_step for value in rotation
        ) + (grip,)

    def summary(self) -> dict:
        with self._lock:
            return {
                "connected": self._connected, "paused": self._paused,
                "gripper": self._gripper, "keys": sorted(self._keys),
                "jog_remaining": self._jog_remaining,
            }


class TeleopServer:
    def __init__(self, state: TeleopState, *, host: str, port: int, preview_fps: float):
        if not 0 < port < 65536 or preview_fps <= 0:
            raise ValueError("invalid port or preview FPS")
        self.state, self.host, self.port = state, host, port
        self.preview_fps = preview_fps
        self.token = secrets.token_urlsafe(24)
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._preview: tuple[int, bytes, bytes, dict] | None = None
        self._sequence = 0
        self._message = "Waiting for browser"

    def publish(self, cam0_jpeg: bytes, wrist_jpeg: bytes, status: dict) -> None:
        with self._lock:
            self._sequence += 1
            self._preview = (self._sequence, cam0_jpeg, wrist_jpeg, dict(status))

    def set_message(self, message: str) -> None:
        with self._lock:
            self._message = message

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="piper-teleop-web", daemon=True)
        self._thread.start()
        if not self._started.wait(15):
            raise TimeoutError("teleop web server did not start")
        if self._error:
            raise RuntimeError("teleop web server failed to start") from self._error

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _authorized(self, request: web.Request) -> bool:
        return secrets.compare_digest(request.query.get("token", ""), self.token)

    async def _index(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            raise web.HTTPForbidden()
        page = Path(__file__).with_name("piper_teleop.html").read_text(encoding="utf-8")
        return web.Response(text=page.replace("__TOKEN__", self.token), content_type="text/html",
                            headers={"Cache-Control": "no-store"})

    async def _websocket(self, request: web.Request) -> web.StreamResponse:
        if not self._authorized(request):
            raise web.HTTPForbidden()
        if not self.state.connect():
            raise web.HTTPConflict(text="another operator is connected")
        ws = web.WebSocketResponse(heartbeat=10, max_msg_size=4096)
        try:
            await ws.prepare(request)
            pump = asyncio.create_task(self._pump(ws))
            try:
                async for message in ws:
                    if message.type != WSMsgType.TEXT:
                        continue
                    try:
                        payload = json.loads(message.data)
                        if not isinstance(payload, dict):
                            raise ValueError("expected a JSON object")
                        if payload.get("type") == "input":
                            self.state.input(payload.get("keys"))
                        elif payload.get("type") == "command":
                            self.state.command(payload.get("name"))
                        elif payload.get("type") == "jog":
                            self.state.jog(payload.get("key"), payload.get("steps"))
                        else:
                            raise ValueError("unknown message type")
                    except (ValueError, TypeError) as exc:
                        await ws.send_json({"type": "error", "message": str(exc)})
            finally:
                pump.cancel()
                await asyncio.gather(pump, return_exceptions=True)
        finally:
            self.state.disconnect()
        return ws

    async def _pump(self, ws: web.WebSocketResponse) -> None:
        last_sequence = -1
        while not ws.closed:
            with self._lock:
                preview, status_message = self._preview, self._message
            if preview and preview[0] != last_sequence:
                sequence, cam0, wrist, status = preview
                await asyncio.wait_for(ws.send_json({
                    "type": "preview", "sequence": sequence,
                    "cam0": base64.b64encode(cam0).decode("ascii"),
                    "wrist": base64.b64encode(wrist).decode("ascii"),
                    "status": status, "message": status_message,
                    "control": self.state.summary(),
                }), timeout=2)
                last_sequence = sequence
            await asyncio.sleep(1 / self.preview_fps)

    async def _serve(self) -> None:
        app = web.Application(client_max_size=4096)
        app.router.add_get("/", self._index)
        app.router.add_get("/ws", self._websocket)
        runner = web.AppRunner(app, access_log=None)
        try:
            await runner.setup()
            site = web.TCPSite(runner, self.host, self.port)
            await site.start()
            self._started.set()
            while not self._stop.is_set():
                await asyncio.sleep(0.1)
        finally:
            await runner.cleanup()

    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as exc:
            self._error = exc
            self._started.set()
