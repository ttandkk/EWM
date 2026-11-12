"""Fixed-center control: no ball observation or launch parameters enter control."""
import numpy as np
from piper_bench.tasks.tennis.oracle import TennisOracle


class FixedCenterOracle(TennisOracle):
    def action(self):
        if self.started is None:
            _, _, tcp = self.runtime.state()
            self.begin('hold', tcp)
            self.hold_yaw = 0.
            self.hold_center = np.array([.45, 0., self.cfg.container_hold_z])
        # Hold stage only levels the passive receiver using its measured pose.
        # It does not call update_prediction or inspect ball state for control.
        return self.container_action()


def run(runtime, task, plan, audit, rollout, resume=False):
    import json
    from piper_bench.collection.protocol import episode_spec
    from piper_bench.collection.protocol import write_json
    path = audit/'fixed_center_diagnostic.json'
    if resume and path.exists() and json.loads(path.read_text()).get('complete'):
        return
    results = []
    for slot in range(plan['episodes']):
        record, _ = rollout(runtime, task, episode_spec(plan, slot),
                            audit/'fixed_center_progress.json', oracle_factory=FixedCenterOracle)
        results.append(record)
        write_json(path, dict(complete=False, results=results))
    write_json(path, dict(complete=True, center_xy_m=[.45, 0.],
        policy='move_to_fixed_center_from_reset_then_level_receiver_without_ball_feedback',
        trials=len(results), successes=sum(bool(r['success']) for r in results),
        plan=plan, results=results, interpretation='diagnostic only; expert acceptance remains the collection gate'))
