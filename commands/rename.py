from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import List

from commands.files_utils import files_from
from commands.files_utils import generate_file_name
from commands.types import FileDescription
from commands.types import StrainInfo


def rename_plate_files(images_folder: Path, strains: List[StrainInfo], extensions: List[str]) -> None:
    left_descriptions = []
    left_files = []
    for extension in extensions:
        descriptions, files = rename_for_extension(images_folder, strains, extension)
        left_descriptions.extend(descriptions)
        left_files.extend(files)

    print_left_items(left_descriptions, description="Can't find files for:")
    print_left_items(left_files, description='Unexpected files:')


def rename_for_extension(images_folder: Path, strains: List[StrainInfo], extension: str):
    files_iterator = iter(files_from(images_folder, extension))
    file_descriptions = iter(generate_descriptions(strains))
    for file, description in zip(files_iterator, file_descriptions):
        new_path = file.parent / generate_file_name(description, extension)
        file.rename(new_path)
        print(f'Rename: {file} -> {new_path}')
    return list(file_descriptions), list(files_iterator)


def generate_descriptions(strains: Iterable[StrainInfo]) -> Iterator[FileDescription]:
    return (
        FileDescription(name=strain.name.replace(' ', ''), number=number, side=side)
        for strain in strains
        for number in range(1, strain.plates_number + 1)
        for side in ('back', 'front')
    )


def print_left_items(iterator: Iterator[Any], description: str) -> None:
    values = list(iterator)
    if values:
        print(description)
        print(*values, sep='\n')
        print(f'Total: {len(values)} elements')
