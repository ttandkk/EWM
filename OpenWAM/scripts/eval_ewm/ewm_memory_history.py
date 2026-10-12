"""Continuous physical-observation history; independent of policy request cadence."""
from collections import OrderedDict
import numpy as np


class RGBEventHistoryBuffer:
    def __init__(self):
        self.frames=OrderedDict()

    def observe(self,rgb,event,frame_id):
        frame_id=int(frame_id)
        if self.frames and frame_id<=next(reversed(self.frames)):
            raise ValueError('history needs a new physical frame; reset at episode boundaries')
        self.frames[frame_id]=(np.array(rgb,copy=True),np.array(event,copy=True))
        while self.frames and next(iter(self.frames))<frame_id-9:
            self.frames.popitem(last=False)
        events=np.zeros((10,*event.shape),dtype=event.dtype)
        event_valid=np.zeros(10,dtype=np.bool_)
        for i,offset in enumerate(range(-9,1)):
            item=self.frames.get(frame_id+offset)
            if item is not None:events[i]=item[1];event_valid[i]=True
        history=np.zeros((3,*rgb.shape),dtype=rgb.dtype)
        rgb_valid=np.array([False,False,False,True],dtype=np.bool_)
        for i,offset in enumerate((-9,-6,-3)):
            item=self.frames.get(frame_id+offset)
            if item is not None:history[i]=item[0];rgb_valid[i]=True
        return dict(memory_rgb=history,memory_event=events,
                    memory_rgb_valid=rgb_valid,memory_event_valid=event_valid)
