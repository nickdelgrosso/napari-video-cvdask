from typing import Tuple, List, Dict, Union

from dask.array import Array

from .cvdask import dask_array_from_filename


def napari_get_reader(path):
    filenames = [path] if isinstance(path, str) else path
    # if we know we cannot read all the files, we immediately return None.
    for filename in filenames:
        if not any([filename.endswith(ext) for ext in [".mp4", ".mov", ".avi"]]):
            return None
    return reader_function


def reader_function(path: Union[str, List[str]]) -> List[Tuple[Array, Dict, str]]:
    """Take a path or list of paths and return a list of LayerData tuples."""
    filenames = [path] if isinstance(path, str) else path
    return [(dask_array_from_filename(filename=filename), {}, "image") for filename in filenames]

