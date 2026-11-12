"""Evidence collected from the running simulator, never from import probes."""
from __future__ import annotations

import json
from pathlib import Path


class RuntimeAudit:
    def __init__(self, env):
        self.env = env
        self.rows = []
        self.previous = {}

    def capture(self):
        import numpy as np
        from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul

        def tensor(value):
            return value.torch if hasattr(value, "torch") else value

        row = {"cameras": {}}
        for name in ("cam0", "cam_wrist"):
            camera = self.env.scene[name]
            data = camera.data
            rgb = tensor(data.output["rgb"])[0, ..., :3].detach().cpu().numpy()
            item = {"shape": list(rgb.shape), "finite": bool(np.isfinite(rgb).all()),
                    "std": float(np.std(rgb)), "position": tensor(data.pos_w)[0].detach().cpu().tolist()}
            item["frame_change"] = (float(np.mean(np.abs(rgb.astype(float) - self.previous[name])))
                                    if name in self.previous else 0.)
            self.previous[name] = rgb.astype(float)
            if name == "cam_wrist":
                # Read the physical hand at capture time, as the rigid sensor does.
                hand = camera._hand_view.get_transforms().to(device=camera.device)
                position = tensor(data.pos_w)
                rotation = tensor(data.quat_w_opengl)
                relative_position = quat_apply_inverse(hand[:, 3:7], position - hand[:, :3])
                relative_rotation = quat_mul(quat_inv(hand[:, 3:7]), rotation)
                actual_pos = relative_position[0].detach().cpu().numpy()
                actual_rot = relative_rotation[0].detach().cpu().numpy()
                expected_pos = np.asarray(camera.cfg.offset.pos)
                expected_rot = np.asarray(camera.cfg.offset.rot)
                item["mount_position_error_m"] = float(np.linalg.norm(actual_pos - expected_pos))
                item["mount_rotation_error"] = float(min(np.linalg.norm(actual_rot - expected_rot),
                                                        np.linalg.norm(actual_rot + expected_rot)))
            row["cameras"][name] = item
        row["joints"] = tensor(self.env.scene["robot"].data.joint_pos)[0].detach().cpu().tolist()
        self.rows.append(row)

    def finish(self, directory: Path, *, events=False):
        import numpy as np
        import omni.usd

        checks = {"frames": len(self.rows) > 1}
        if self.rows:
            joints = np.asarray([r["joints"] for r in self.rows])
            checks["finite_joints"] = bool(np.isfinite(joints).all())
            checks["robot_moves"] = bool(np.max(np.ptp(joints[:, :6], axis=0)) > .01)
            for name in ("cam0", "cam_wrist"):
                values = [r["cameras"][name] for r in self.rows]
                checks[name + "_rgb"] = all(v["shape"] == [256, 256, 3] and v["finite"] and v["std"] > 1. for v in values)
                checks[name + "_changes"] = max(v["frame_change"] for v in values) > .01
                if name == "cam_wrist":
                    positions = np.asarray([v["position"] for v in values])
                    checks["wrist_moves"] = bool(np.max(np.ptp(positions, axis=0)) > .001)
                    checks["wrist_rigid_mount"] = all(v["mount_position_error_m"] < 1e-4 and v["mount_rotation_error"] < 1e-3 for v in values)
        stage = omni.usd.get_context().get_stage()
        errors = [str(e) for e in stage.GetCompositionErrors()]
        checks["usd_composition"] = not errors
        report = {"passed": all(checks.values()), "checks": checks, "frames": len(self.rows),
                  "usd_composition_errors": errors,
                  "used_usd_layers": [layer.identifier for layer in stage.GetUsedLayers()],
                  "texture_dependency_closure_hashed": False,
                  "samples": self.rows}
        entry = Path(self.env.cfg.scene.robot.spawn.usd_path)
        derivation = entry.parent / "derivation.json"
        if derivation.is_file():
            report["piper_usd_derivation"] = {"entry": str(entry), "manifest": str(derivation),
                                               "record": json.loads(derivation.read_text())}
        if events:
            from .evis import validate_recording
            report["events"] = validate_recording(directory)
        (directory / "deployment_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report
