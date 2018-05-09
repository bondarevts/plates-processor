from pathlib import Path
from typing import Iterable


def prepare_file_name(name: str) -> Path:
    return Path(name.replace('/', ':'))


def files_from(folder: Path, extension: str) -> Iterable[Path]:
    return sorted(
        file
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() == extension
    )
