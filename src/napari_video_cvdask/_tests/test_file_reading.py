from glob import glob
from pytest import mark
from unittest.mock import Mock, patch

from napari_video_cvdask.cvdask import dask_array_from_filename, to_dask_array
from dask.array import Array
import cv2

cases = glob("./data/file_example_*")

@mark.parametrize("filename", cases)
def test_testdata_reads(filename):
    data = dask_array_from_filename(filename)    
    assert isinstance(data, Array)



def test_reader_interprets_videocapture():
    
    def side_effect(arg, *args, **kwargs):
        props = {
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FRAME_WIDTH: 720,
            cv2.CAP_PROP_FRAME_COUNT: 40
        }
        return props[arg]

    cap = Mock(cv2.VideoCapture)
    cap.get.side_effect = side_effect

    data = to_dask_array(cap)

    assert isinstance(data, Array)
    assert data.shape == (40, 1080, 720, 3)

