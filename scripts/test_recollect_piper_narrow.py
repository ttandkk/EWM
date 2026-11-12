import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

spec=importlib.util.spec_from_file_location('narrow_recollect',Path(__file__).with_name('recollect_piper_narrow.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def setup_case(tmp_path,monkeypatch,passed=True):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'TASKS',('moving_carton',))
    run=tmp_path/'run';run.mkdir();(run/'logs').mkdir()
    conveyor=tmp_path/'datasets/piper_conveyor';conveyor.mkdir(parents=True);(conveyor/'keep').write_text('untouched')
    old=tmp_path/'datasets/piper_moving_carton';old.mkdir();(old/'old').write_text('old data')
    stage=run/'staging/piper_moving_carton'
    m.write(run/'replacement_plan.json',dict(conveyor_before=m.fingerprint(conveyor)))
    def execute(command,*args):
        if command[0]=='collect':
            stage.mkdir(parents=True);(stage/'.export_complete').write_text('{}')
            m.write(stage/'meta/piper_validation.json',dict(passed=True,episodes=50,frames=1234,raw_events_validated=True))
        return 0
    module=SimpleNamespace(verify_source=lambda x:None,commands=lambda *a,**k:(['collect'],['validate']),execute=execute,
        acceptance=lambda *a:dict(passed=passed,trials=20,successes=20 if passed else 0))
    monkeypatch.setattr(m,'load_launcher',lambda *a:module)
    monkeypatch.setattr(m,'launcher_args',lambda *a:None)
    for n in m.CACHE_NAMES:
        c=tmp_path/'tmp/training_data_cache'/n;c.mkdir(parents=True);(c/'old').write_text('cache')
    return run,old,conveyor


def test_validated_replacement_removes_only_old_task_and_caches(tmp_path,monkeypatch):
    run,old,conveyor=setup_case(tmp_path,monkeypatch)
    before=m.fingerprint(conveyor);m.collect(run)
    assert (old/'.export_complete').exists() and not (old/'old').exists()
    assert m.fingerprint(conveyor)==before
    assert not (tmp_path/'datasets/.narrow_recollection_in_progress.json').exists()
    assert json.loads((run/'collection_complete.json').read_text())['passed']
    assert all(not (tmp_path/'tmp/training_data_cache'/n).exists() for n in m.CACHE_NAMES)
    m.collect(run)  # Resume skips already published task and does not delete it.
    assert (old/'.export_complete').exists()


def test_failed_gate_keeps_old_data_and_blocks_cache(tmp_path,monkeypatch):
    run,old,_=setup_case(tmp_path,monkeypatch,False)
    with pytest.raises(RuntimeError):m.collect(run)
    assert (old/'old').read_text()=='old data'
    assert not (run/'collection_complete.json').exists()
    assert (tmp_path/'datasets/.narrow_recollection_in_progress.json').exists()
    assert all((tmp_path/'tmp/training_data_cache'/n/'old').exists() for n in m.CACHE_NAMES)


def test_delete_rejects_symlinks_and_unapproved_parent(tmp_path):
    target=tmp_path/'keep';target.mkdir();link=tmp_path/'link';link.symlink_to(target,target_is_directory=True)
    with pytest.raises(ValueError):m.remove_directory(link,tmp_path)
    with pytest.raises(ValueError):m.remove_directory(target,tmp_path/'different')
    assert target.exists()
