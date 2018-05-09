from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import List

from commands.files_utils import files_from
from commands.files_utils import prepare_file_name
from commands.types import FileDescription
from commands.types import StrainInfo


def rename_plate_files(images_folder: Path, strains: List[StrainInfo], extension: str) -> None:
    files_iterator = iter(files_from(images_folder, extension))
    file_descriptions = iter(generate_descriptions(strains))

    for file, description in zip(files_iterator, file_descriptions):
        new_filename = prepare_file_name(f'{description.name}_{description.number}_{description.side}{extension}')
        new_path = file.parent / new_filename
        file.rename(new_path)
        print(f'Rename: {file} -> {new_path}')

    print_left_items(file_descriptions, description="Can't find files for:")
    print_left_items(files_iterator, description='Unexpected files:')


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
