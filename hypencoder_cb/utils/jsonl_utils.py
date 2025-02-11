import pathlib
from contextlib import contextmanager

import jsonlines

from hypencoder_cb.utils.io_utils import resolve_path


@contextmanager
def JsonlReader(filepath: str, **kwargs):
    filepath = resolve_path(filepath)
    with jsonlines.open(filepath, **kwargs) as reader:
        yield reader


@contextmanager
def JsonlWriter(
    filepath: str, check_exists: bool = False, mode: str = "w", **kwargs
):
    filepath = resolve_path(filepath)

    path_object = pathlib.Path(filepath)
    if check_exists and path_object.exists():
        raise FileExistsError(f"File {filepath} already exists.")

    if not path_object.parent.exists():
        path_object.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(filepath, mode=mode, **kwargs) as writer:
        yield writer
