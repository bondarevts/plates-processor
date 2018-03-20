import csv
import sys
from collections import namedtuple
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Union

FileDescription = namedtuple('ImageDescription', 'name number side')
StrainInfo = namedtuple('StrainInfo', 'name plates_number')


def get_names(strain_descriptions_file: Union[Path, str]) -> List[StrainInfo]:
    with open(strain_descriptions_file, newline='') as csv_file:
        next(csv_file, None)

        return [StrainInfo(name=record[1], plates_number=int(record[2]))
                for record in csv.reader(csv_file, dialect='excel')]


def files_from(folder: Path, extension: str) -> Iterator[Path]:
    return iter(sorted(
        file
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() == extension
    ))


def generate_descriptions(strains: Iterable[StrainInfo]) -> Iterator[FileDescription]:
    return (
        FileDescription(name=strain.name, number=number, side=side)
        for strain in strains
        for number in range(1, strain.plates_number + 1)
        for side in ('back', 'front')
    )


def print_left_items(iterator: Iterator[Any], description: str) -> None:
    values = list(iterator)
    if values:
        print(description)
        print(*values, sep='\n')


def rename_plate_files(images_folder: Path, strains: List[StrainInfo], extension: str) -> None:
    files_iterator = files_from(images_folder, extension)
    file_descriptions = generate_descriptions(strains)

    for file, description in zip(files_iterator, file_descriptions):
        new_filename = f'{description.name}_{description.number}_{description.side}{extension}'.replace('/', ':')
        new_path = file.parent / new_filename
        file.rename(new_path)
        print(f'Rename: {file} -> {new_path}')

    print_left_items(file_descriptions, description="Can't find files for:")
    print_left_items(files_iterator, description='Unexpected files:')


def rename(extension: str, names_file: str, input_folder: str) -> None:
    extension = extension.lower()
    if extension and not extension.startswith('.'):
        extension = '.' + extension
    rename_plate_files(
        images_folder=Path(input_folder),
        strains=get_names(names_file),
        extension=extension
    )


def print_usage() -> None:
    print(f'Usage: {sys.argv[0]} rename <extension> <names_file> <input_folder>')


def main() -> None:
    if len(sys.argv) == 1:
        print_usage()
        return

    *_, extension, names_file, input_folder = sys.argv
    rename(extension, names_file, input_folder)


if __name__ == '__main__':
    main()
