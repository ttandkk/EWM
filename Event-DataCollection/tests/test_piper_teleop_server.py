"""No-GPU checks for the Piper browser teleoperation protocol."""
from __future__ import annotations

import asyncio
from pathlib import Path
import socket
import sys
import time
import unittest

from aiohttp import ClientSession, WSServerHandshakeError
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from piper_teleop_server import TeleopServer, TeleopState
from piper_teleop_geometry import wrist_planar_delta


class TeleopStateTest(unittest.TestCase):
    def test_input_pause_gripper_and_disconnect(self):
        state = TeleopState(pos_step=0.02, rot_step=0.04, timeout_s=0.05)
        self.assertIsNone(state.action())
        self.assertTrue(state.connect())
        self.assertFalse(state.connect())
        state.input(["w", "d", "v"])
        state.command("resume")
        action = state.action()
        self.assertAlmostEqual(action[0], 0.02 / 2**0.5)
        self.assertAlmostEqual(action[1], 0.02 / 2**0.5)
        self.assertEqual(action[5], 0.04)
        self.assertEqual(action[6], 1.0)
        state.command("grip")
        self.assertEqual(state.action()[6], -1.0)
        state.command("pause")
        self.assertIsNone(state.action())
        state.command("resume")
        state.input([])
        self.assertEqual(state.action()[:6], (0., 0., 0., 0., 0., 0.))
        time.sleep(0.07)
        self.assertIsNone(state.action())
        self.assertTrue(state.summary()["paused"])
        state.disconnect()
        self.assertIsNone(state.action())

    def test_rejects_invalid_commands_and_one_shot_queue(self):
        with self.assertRaises(ValueError):
            TeleopState(pos_step=1.0, rot_step=0.01)
        state = TeleopState(pos_step=0.01, rot_step=0.01)
        state.connect()
        with self.assertRaises(ValueError):
            state.input(["invalid"])
        with self.assertRaises(ValueError):
            state.command("delete")
        state.command("discard")
        self.assertEqual(state.take_commands(), ["discard"])
        self.assertEqual(state.take_commands(), [])
        self.assertIsNone(state.action())

    def test_bounded_jog_then_auto_pause(self):
        state = TeleopState(pos_step=0.02, rot_step=0.04, timeout_s=0.05)
        state.connect()
        for invalid in (0, 9, True):
            with self.assertRaises(ValueError):
                state.jog("w", invalid)
        state.jog("w", 4)
        time.sleep(0.07)  # A bounded jog is independent of browser heartbeat.
        for _ in range(4):
            self.assertAlmostEqual(state.action()[0], 0.02)
        self.assertTrue(state.summary()["paused"])
        self.assertIsNone(state.action())
        state.jog("d", 8)
        state.command("pause")
        self.assertIsNone(state.action())


class WristGeometryTest(unittest.TestCase):
    def test_image_axes_follow_camera_heading(self):
        np.testing.assert_allclose(wrist_planar_delta([1, 0, -1], 1, 0), [1, 0], atol=1e-6)
        np.testing.assert_allclose(wrist_planar_delta([1, 0, -1], 0, 1), [0, -1], atol=1e-6)
        np.testing.assert_allclose(wrist_planar_delta([0, 1, -1], 1, 0), [0, 1], atol=1e-6)
        np.testing.assert_allclose(wrist_planar_delta([0, 1, -1], 0, 1), [1, 0], atol=1e-6)
        self.assertLessEqual(np.linalg.norm(wrist_planar_delta([1, 1, -1], 1, 1)), 2**0.5 + 1e-6)


class TeleopServerTest(unittest.TestCase):
    def test_authorization_browser_and_preview(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        state = TeleopState(pos_step=0.01, rot_step=0.01)
        server = TeleopServer(state, host="127.0.0.1", port=port, preview_fps=20)
        server.start()
        try:
            asyncio.run(self._exercise(server, state))
        finally:
            server.stop()

    async def _exercise(self, server, state):
        root = f"http://127.0.0.1:{server.port}"
        async with ClientSession() as client:
            async with client.get(root + "/") as response:
                self.assertEqual(response.status, 403)
            async with client.get(root + f"/?token={server.token}") as response:
                self.assertEqual(response.status, 200)
                page = await response.text()
                self.assertIn("Piper", page)
                self.assertIn('data-command="resume"', page)
                self.assertIn('button.dataset.command', page)
                self.assertNotIn("onclick=", page)
            with self.assertRaises(WSServerHandshakeError):
                await client.ws_connect(root + "/ws")
            async with client.ws_connect(root + f"/ws?token={server.token}") as ws:
                server.publish(b"camera0", b"wrist", {"step": 0})
                reply = await ws.receive_json(timeout=2)
                self.assertEqual(reply["type"], "preview")
                self.assertEqual(reply["status"]["step"], 0)
                await ws.send_json({"type": "input", "keys": ["w"]})
                await ws.send_json({"type": "command", "name": "resume"})
                await asyncio.sleep(0.05)
                self.assertEqual(state.action()[0], 0.01)
                await ws.send_json({"type": "jog", "key": "d", "steps": 2})
                await asyncio.sleep(0.05)
                self.assertEqual(state.action()[1], 0.01)
                self.assertEqual(state.action()[1], 0.01)
                self.assertIsNone(state.action())
            for _ in range(20):
                if state.action() is None:
                    break
                await asyncio.sleep(0.01)
            self.assertIsNone(state.action())


if __name__ == "__main__":
    unittest.main()
