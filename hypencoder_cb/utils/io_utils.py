import pathlib


def resolve_path(path: str) -> str:
    return str(pathlib.Path(path).expanduser().resolve())
