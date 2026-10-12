import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from piper_conveyor_resume import prepare_resume


def make_export(root):
    meta = root / 'meta'
    meta.mkdir()
    (meta / 'info.json').write_text(json.dumps({'total_episodes': 2}))
    (meta / 'piper_collection_config.json').write_text(json.dumps({
        'requested_successful_episodes': 2, 'seed': 12, 'oracle_profile': 'frozen'}))
    attempts = [{'seed': 12, 'saved_episode': 0}, {'seed': 13, 'success': False},
                {'seed': 14, 'saved_episode': 1}]
    (meta / 'piper_conveyor_attempts.json').write_text(json.dumps(attempts))
    for path in (root / '.collection_complete', root / '.export_complete',
                 meta / 'piper_validation.json', meta / 'ewm_reader_smoke.json',
                 meta / 'piper_completion_audit.json'):
        path.write_text('old completed export')
    return attempts


def test_append_invalidates_stale_completion_and_preserves_seed_history(tmp_path):
    attempts = make_export(tmp_path)
    original = (tmp_path / 'meta/piper_conveyor_attempts.json').read_bytes()
    assert prepare_resume(tmp_path, 4) == (attempts, 2)
    assert (tmp_path / 'meta/piper_conveyor_attempts.json').read_bytes() == original
    config = json.loads((tmp_path / 'meta/piper_collection_config.json').read_text())
    assert config == {'requested_successful_episodes': 4, 'seed': 12, 'oracle_profile': 'frozen'}
    assert not (tmp_path / '.collection_complete').exists()
    assert not (tmp_path / '.export_complete').exists()
    assert not (tmp_path / 'meta/piper_validation.json').exists()
    history = json.loads((tmp_path / 'meta/piper_resume_history.json').read_text())
    assert history == [{'saved_episodes': 2, 'completed_attempts': 3,
                        'previous_target': 2, 'target': 4}]


@pytest.mark.parametrize('target,broken', [(1, False), (4, True)])
def test_reject_inconsistent_or_smaller_resume_before_mutation(tmp_path, target, broken):
    make_export(tmp_path)
    if broken:
        (tmp_path / 'meta/info.json').write_text('{"total_episodes": 3}')
    with pytest.raises(ValueError):
        prepare_resume(tmp_path, target)
    assert (tmp_path / '.export_complete').exists()
    assert not (tmp_path / 'meta/piper_resume_history.json').exists()
