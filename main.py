#!/usr/bin/env python
import csv
import math
import sys
from collections import namedtuple
from itertools import groupby
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Union

import cv2 as cv
import numpy as np
from PIL import Image

HSV_MIN = np.array((0, 0, 75), np.uint8)
HSV_MAX = np.array((255, 255, 214), np.uint8)

CvImage = np.ndarray
CvContour = np.ndarray

FileDescription = namedtuple('ImageDescription', 'name number side')
StrainInfo = namedtuple('StrainInfo', 'name plates_number')
FilesGroup = namedtuple('FilesGroup', 'name side files')
ImageSize = namedtuple('ImageSize', 'width height')
Layout = namedtuple('Layout', 'rows columns')


def prepare_extension(extension: str) -> str:
    extension = extension.lower()
    if extension and not extension.startswith('.'):
        extension = '.' + extension
    return extension


def get_names(strain_descriptions_file: Union[Path, str]) -> List[StrainInfo]:
    with open(strain_descriptions_file, newline='') as csv_file:
        next(csv_file, None)

        return [StrainInfo(name=record[1], plates_number=int(record[2]))
                for record in csv.reader(csv_file, dialect='excel')]


def files_from(folder: Path, extension: str) -> Iterable[Path]:
    return sorted(
        file
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() == extension
    )


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


def prepare_file_name(name: str) -> Path:
    return Path(name.replace('/', ':'))


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


def rename(extension: str, names_file: str, input_folder: str) -> None:
    rename_plate_files(
        images_folder=Path(input_folder),
        strains=get_names(names_file),
        extension=prepare_extension(extension),
    )


def extract_groups(files: Iterable[Path]) -> Iterable[FilesGroup]:
    def group_key(file):
        return file.name.split('_')[0], file.stem.split('_')[-1]

    groups = groupby(sorted(files, key=group_key), key=group_key)
    return (FilesGroup(name=name, side=side, files=sorted(group_files, key=lambda file: int(file.name.split('_')[1])))
            for (name, side), group_files in groups)


def image_size(image_path: Path) -> ImageSize:
    with Image.open(image_path) as image:
        return ImageSize(*image.size)


def get_layout(files_number: int) -> Layout:
    rows = int(files_number ** 0.5)
    columns = math.ceil(files_number / rows)
    return Layout(rows=rows, columns=columns)


def combine_files_in_group(group: FilesGroup, target_folder: Path, extension: str) -> None:
    print(f'Combine {group.name}, {group.side}. ', end='')
    sizes = list(map(image_size, group.files))
    max_width = max(size.width for size in sizes)
    max_height = max(size.height for size in sizes)
    files_number = len(group.files)

    layout = get_layout(files_number)

    target_height = layout.rows * max_height
    target_width = layout.columns * max_width

    print(f'Max width: {max_width}, max height: {max_height}; '
          f'images in group: {files_number}; '
          f'layout: {layout.rows}x{layout.columns}; '
          f'target size: {target_width}, {target_height}')

    target_image = Image.new('RGB', (target_width, target_height))
    x_offset = y_offset = 0
    for i, file in enumerate(group.files):
        with Image.open(file) as image:
            target_image.paste(image, (x_offset, y_offset))
        x_offset += max_width
        if (i + 1) % layout.columns == 0:
            x_offset = 0
            y_offset += max_height

    target_image.save(target_folder / prepare_file_name(f'{group.name}_{group.side}{extension}'))


def combine_strain_files(extension: str, input_folder: Path):
    files = files_from(input_folder, extension)
    target_folder = input_folder / 'combined'
    target_folder.mkdir(exist_ok=True)
    for group in extract_groups(files):
        combine_files_in_group(group, target_folder, extension)


def combine(extension: str, input_folder: str) -> None:
    combine_strain_files(
        extension=prepare_extension(extension),
        input_folder=Path(input_folder),
    )


def convert_to_black_and_white(img: CvImage) -> CvImage:
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    return cv.inRange(hsv, HSV_MIN, HSV_MAX)


def get_complex_contours(image: CvImage) -> Iterable[CvContour]:
    threshold = convert_to_black_and_white(image)

    _, contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    hull_contours = map(cv.convexHull, contours)
    complex_shape_vertices_number = 100
    return (c for c in hull_contours if len(c) > complex_shape_vertices_number)


def get_plate_contour(contours: Iterable[CvContour]) -> CvContour:
    *_, plate_contour = max((cv.contourArea(contour), i, contour)
                            for i, contour in enumerate(contours))
    return plate_contour


def fill_black_outside_contour(image: CvImage, plate_contour: CvContour) -> None:
    mask = np.zeros_like(image)
    (x, y), radius = cv.minEnclosingCircle(plate_contour)
    center = (int(x), int(y))
    radius = int(radius)
    cv.circle(mask, center, radius, color=(255, 255, 255), thickness=-1)
    image[mask != 255] = 0


def crop_plate_square(image: CvImage, plate_contour: CvContour) -> CvImage:
    fill_black_outside_contour(image, plate_contour)
    x, y, w, h = cv.boundingRect(plate_contour)
    size = max(w, h)
    return image[y:y + size, x:x + size]


def extract_plate(image_path: Path) -> CvImage:
    image = cv.imread(image_path.as_posix())
    contours = get_complex_contours(image)
    plate_contour = get_plate_contour(contours)
    return crop_plate_square(image, plate_contour)


def crop_files(extension: str, input_folder: Path) -> None:
    files = files_from(input_folder, extension)
    for file in files:
        print(f'Crop {file}', end='; ')
        image = extract_plate(file)

        target_folder = file.parent / 'plates'
        target_folder.mkdir(exist_ok=True)
        target_path = target_folder / prepare_file_name(f'{file.stem}-plate{file.suffix}')
        print(f'saved in {target_path}')
        cv.imwrite(target_path.as_posix(), image)


def crop(extension: str, input_folder: str) -> None:
    crop_files(
        extension=prepare_extension(extension),
        input_folder=Path(input_folder),
    )


def print_usage() -> None:
    print('Usage: ')
    print(f'\t{sys.argv[0]} rename <extension> <names_file> <input_folder>')
    print(f'\t{sys.argv[0]} combine <extension> <input_folder>')
    print(f'\t{sys.argv[0]} crop <extension> <input_folder>')


def main() -> None:
    if len(sys.argv) == 1:
        print_usage()
        return

    _, command, *args = sys.argv

    if command == 'rename':
        extension, names_file, input_folder = args
        rename(extension, names_file, input_folder)

    elif command == 'combine':
        extension, input_folder = args
        combine(extension, input_folder)

    elif command == 'crop':
        extension, input_folder = args
        crop(extension, input_folder)

    else:
        print(f'Unexpected command: {command}')
        print_usage()


if __name__ == '__main__':
    main()
