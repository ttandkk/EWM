"""Invalidate completed-export metadata before appending compatible episodes."""
import json
from pathlib import Path


def prepare_resume(output: Path, target: int):
    meta = output / 'meta'
    config_path = meta / 'piper_collection_config.json'
    config = json.loads(config_path.read_text())
    attempts = json.loads((meta / 'piper_conveyor_attempts.json').read_text())
    saved = int(json.loads((meta / 'info.json').read_text())['total_episodes'])
    indices = [record['saved_episode'] for record in attempts if 'saved_episode' in record]
    if indices != list(range(saved)):
        raise ValueError('saved episode metadata and attempt history disagree')
    if target < saved:
        raise ValueError('resume target cannot be smaller than the existing episode count')
    history_path = meta / 'piper_resume_history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(dict(saved_episodes=saved, completed_attempts=len(attempts),
                        previous_target=config['requested_successful_episodes'], target=target))
    history_path.write_text(json.dumps(history, indent=2) + '\n')
    # Readers must never interpret a previous completed export as the growing dataset.
    for path in (output / '.collection_complete', output / '.export_complete',
                 meta / 'piper_validation.json', meta / 'ewm_reader_smoke.json',
                 meta / 'piper_completion_audit.json'):
        path.unlink(missing_ok=True)
    config['requested_successful_episodes'] = target
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    return attempts, saved
