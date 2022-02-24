from threading import Lock
from pathlib import Path
from typing import Union

import numpy as np
import dask
import dask.array as da
import cv2


def dask_array_from_filename(filename: Union[str, Path], **kwargs) -> da.Array:
    """Wrapper around to_dask_array()"""
    cap=cv2.VideoCapture(str(filename))
    array = to_dask_array(cap, **kwargs)
    return array


def to_dask_array(cap: cv2.VideoCapture, n_channels = 3, dtype=np.uint8) -> da.Array:
    """Returns a Dask Array from an OpenCV2 VideoCapture object."""
    frame_shape = (get_height(cap), get_width(cap), n_channels)
    n_frames = get_nframes(cap)
    delayeds = [dask.delayed(get_frame)(cap, idx) for idx in range(n_frames)]
    arrays = [da.from_delayed(d, shape=frame_shape, dtype=dtype) for d in delayeds]
    array = da.stack(arrays)
    return array


def get_frame(cap: cv2.VideoCapture, idx: int, *, lock: Lock = Lock()) -> np.ndarray:
    with lock:  # prevent anything else from happening during reading
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        valid, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def get_height(cap: cv2.VideoCapture) -> int:
    return int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    
def get_width(cap: cv2.VideoCapture) -> int:
    return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


def get_nframes(cap: cv2.VideoCapture) -> int:
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))



    
