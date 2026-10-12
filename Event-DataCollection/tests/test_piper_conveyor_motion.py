"""Kinematic gates for smoothly connected Cartesian references."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from piper_conveyor_motion import bezier_motion,line_motion


@pytest.mark.parametrize('duration',[.8,1.8])
def test_reference_preserves_start_velocity_and_stops_without_acceleration_jump(duration):
    start=np.array([.33,-.1,.05]);end=np.array([.155,.17,.1]);velocity=np.array([0.,.06,0.])
    c1=start.copy();c1[2]=.25;c2=end.copy();c2[2]=.25
    def sample(t):return bezier_motion(start,c1,c2,end,t,duration,velocity)
    h=1e-4
    np.testing.assert_allclose(sample(0),start,atol=1e-12)
    np.testing.assert_allclose(sample(duration),end,atol=1e-12)
    np.testing.assert_allclose((-3*sample(0)+4*sample(h)-sample(2*h))/(2*h),velocity,atol=2e-6)
    np.testing.assert_allclose((3*sample(duration)-4*sample(duration-h)+sample(duration-2*h))/(2*h),0,atol=2e-6)
    assert np.linalg.norm((sample(2*h)-2*sample(h)+sample(0))/(h*h))<.03
    assert np.linalg.norm((sample(duration)-2*sample(duration-h)+sample(duration-2*h))/(h*h))<.03


def test_carry_clears_belt_before_lateral_transfer_and_rim_before_lowering():
    start=np.array([.33,-.1,.05]);end=np.array([.155,.17,.1]);c1=start.copy();c1[2]=.25;c2=end.copy();c2[2]=.25
    points=np.stack([bezier_motion(start,c1,c2,end,t,1.8,np.array([0.,.06,0.])) for t in np.linspace(0,1.8,1001)])
    moved=np.abs(points[:,0]-start[0])>.01
    assert np.all(points[moved,2]>.10-.000001)
    assert points[:,2].max()<.25
    reference=line_motion(start,end,1.,1.)
    np.testing.assert_allclose(reference,end,atol=1e-12)
