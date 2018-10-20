from pathlib import Path
from typing import Iterable

from commands.types import FileDescription


def generate_file_name(description: FileDescription, extension: str) -> Path:
    return prepare_file_name(f'{description.name}_{description.number}_{description.side}{extension}')


def prepare_file_name(name: str) -> Path:
    return Path(name.replace('/', ':'))


def parse_file_name(file: Path) -> FileDescription:
    strain_name, number, side = file.name.rsplit('_', maxsplit=2)
    return FileDescription(strain_name, int(number), side)


def files_from(folder: Path, extension: str) -> Iterable[Path]:
    return sorted(
        file
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() == extension
    )
