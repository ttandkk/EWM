"""Continuous Cartesian references for physical conveyor manipulation."""
import numpy as np


def quintic(u):
    u=np.clip(u,0.,1.)
    return u*u*u*(10.+u*(-15.+6.*u))


def bezier_motion(start,control1,control2,end,elapsed,duration,start_velocity=None):
    """C2 start/end easing, preserving a moving interception's initial velocity."""
    u=float(np.clip(elapsed/duration,0.,1.));s=quintic(u)
    p=(1-s)**3*np.asarray(start)+3*(1-s)**2*s*np.asarray(control1)+3*(1-s)*s*s*np.asarray(control2)+s**3*np.asarray(end)
    if start_velocity is not None:
        g=u-6*u**3+8*u**4-3*u**5
        p=p+duration*g*np.asarray(start_velocity)
    return p


def line_motion(start,end,elapsed,duration,start_velocity=None):
    start,end=np.asarray(start),np.asarray(end)
    return bezier_motion(start,start+(end-start)/3,start+2*(end-start)/3,end,elapsed,duration,start_velocity)
