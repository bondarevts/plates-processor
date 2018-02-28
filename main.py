import csv
import itertools
from pathlib import Path
from typing import List


IMAGES_FOLDER = Path('plates-test')
STRAINS_DESCRIPTION_CSV = Path('plates-example.csv')


def get_names(strain_descriptions_file: Path) -> List[str]:
    with open(strain_descriptions_file, newline='') as csv_file:
        next(csv_file, None)

        return [record[1] for record in csv.reader(csv_file, dialect='excel')]


def rename_plates(names: List[str], images_folder: Path, plates_in_test: int) -> None:
    raw_files: List[Path] = sorted(
        file
        for file in images_folder.iterdir()
        if file.is_file() and file.name.endswith('.ARW')
    )
    picture_descriptions = itertools.product(
        names,
        ('back', 'front'),
        range(1, plates_in_test + 1)
    )

    for raw_file, (strain, side, number) in zip(raw_files, picture_descriptions):
        if not raw_file.name.startswith('DSC'):
            print(f'Skipped {raw_file}: Unexpected name. Should starts with DSC')
            continue
        new_filename = f'{strain}_{number}_{side}.ARW'.replace('/', ':')
        new_path = raw_file.parent / new_filename
        raw_file.rename(new_path)
        print(f'{raw_file} -> {new_path}')


def main():
    rename_plates(
        get_names(STRAINS_DESCRIPTION_CSV),
        images_folder=IMAGES_FOLDER,
        plates_in_test=3
    )


if __name__ == '__main__':
    main()
