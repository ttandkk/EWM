"""Threaded WebSocket I/O with serialized inference on the model's owner thread."""
from __future__ import annotations

from concurrent.futures import Future
import json
from pathlib import Path
from queue import Empty, Queue
import threading

from piper_bench.evaluation.wire import MAX_MESSAGE, decode_message, encode_message
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from .metadata import hello_metadata


def serve_policy(policy, *, ready_file, host="127.0.0.1", port=0):
    ready = Path(ready_file).expanduser().resolve()
    ready.parent.mkdir(parents=True, exist_ok=True)
    if ready.exists():
        raise FileExistsError(f"refusing to replace readiness file: {ready}")
    requests = Queue(maxsize=32)
    stopped = threading.Event()
    metadata = hello_metadata(policy.descriptor, getattr(policy, "optimization", None))

    def handler(connection):
        try:
            while not stopped.is_set():
                request, arrays = decode_message(connection.recv(timeout=600))
                result = Future()
                requests.put((request, arrays, result), timeout=10)
                reply = result.result(timeout=600)
                connection.send(reply)
                if request.get("op") == "shutdown":
                    break
        except ConnectionClosed:
            pass
        except Exception as exc:
            try:
                connection.send(encode_message({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
            except ConnectionClosed:
                pass

    with serve(handler, host, port, max_size=MAX_MESSAGE, compression=None, ping_interval=None) as server:
        listener = threading.Thread(target=server.serve_forever, name="policy-websocket", daemon=True)
        listener.start()
        created = False
        try:
            url = f"ws://{host}:{server.socket.getsockname()[1]}"
            # Exclusive creation protects an independently running service.
            with ready.open("x") as stream:
                created = True
                stream.write(json.dumps({"url": url, **metadata}) + "\n")
            print(f"Policy service ready: {url}", flush=True)
            while not stopped.is_set():
                try:
                    request, arrays, result = requests.get(timeout=.5)
                except Empty:
                    if not listener.is_alive():
                        raise RuntimeError("policy listener stopped unexpectedly")
                    continue
                try:
                    if request.get("op") == "shutdown":
                        result.set_result(encode_message({"ok": True}))
                        stopped.set()
                    elif request.get("op") == "hello":
                        result.set_result(encode_message({"ok": True, **metadata}))
                    else:
                        reply, output = policy.handle(request, arrays)
                        result.set_result(encode_message(reply, output))
                except Exception as exc:
                    result.set_result(encode_message({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        finally:
            stopped.set()
            while True:
                try:
                    _, _, result = requests.get_nowait()
                except Empty:
                    break
                result.set_result(encode_message({"ok": False, "error": "policy service is stopping"}))
            server.shutdown()
            listener.join(timeout=10)
            if created:
                ready.unlink(missing_ok=True)
