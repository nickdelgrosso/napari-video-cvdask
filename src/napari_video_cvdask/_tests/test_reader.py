from glob import glob

from pytest import mark
import numpy as np

from napari_video_cvdask import napari_get_reader


# tmp_path is a pytest fixture
cases = glob("./data/file_example_*")


@mark.parametrize("filename", cases)
def test_reader(filename):
    """An example of how you might test your plugin."""

    # try to read it back in
    reader = napari_get_reader(filename)
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(filename)
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) > 0

    data, metadata, layer_type = layer_data_tuple
    assert len(data.shape) == 4  
    assert layer_type == 'image'

    
def test_get_reader_pass():
    reader = napari_get_reader("fake.file")
    assert reader is None
