"""Conveyor dataset writer using the existing Piper LeRobot protocol and codecs."""
from piper_conveyor import PROMPT, TASK
from write_piper_stack_lerobot import features as stack_features, main


def features():
    result = stack_features()
    result.pop('observation.cube_pose')
    for name, count in (('object_pose',14), ('object_velocity',12), ('object_active',2), ('sim_time',1)):
        result['observation.'+name] = {'dtype':'float32', 'shape':(count,), 'names':None}
    return result


if __name__ == '__main__':
    main(task_prompt=PROMPT, feature_builder=features,
         contract_overrides={'task_id':TASK, 'object_count':2,
                             'simulation_clock':'fixed_simulation_time_inference_paused',
                             'physics_hz':120, 'physics_backend':'CPU PhysX, GPU RTX rendering'})
