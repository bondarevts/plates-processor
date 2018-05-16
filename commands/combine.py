import math
import traceback
from itertools import groupby
from pathlib import Path
from typing import Iterable

import cv2 as cv
import numpy as np

from commands.files_utils import files_from
from commands.files_utils import prepare_file_name
from commands.image_utils import convert_to_black_and_white
from commands.image_utils import image_center
from commands.image_utils import image_size
from commands.types import ColorRange
from commands.types import CvContour
from commands.types import CvImage
from commands.types import FilesGroup
from commands.types import Layout
from commands.types import Point

TEXT_HSV_RANGE = ColorRange(
    min=np.array((0, 0, 4), np.uint8),
    max=np.array((114, 192, 72), np.uint8)
)


def combine_strain_files(extension: str, input_folder: Path, target_folder: Path, with_rotation: bool) -> None:
    files = files_from(input_folder, extension)
    for group in extract_groups(files):
        combine_files_in_group(group, target_folder, extension, with_rotation)


def extract_groups(files: Iterable[Path]) -> Iterable[FilesGroup]:
    def group_key(file):
        return file.name.split('_')[0], file.stem.split('_')[-1]

    groups = groupby(sorted(files, key=group_key), key=group_key)
    return (FilesGroup(name=name, side=side, files=sorted(group_files, key=lambda file: int(file.name.split('_')[1])))
            for (name, side), group_files in groups)


def combine_files_in_group(group: FilesGroup, target_folder: Path, extension: str, with_rotation: bool) -> None:
    print(f'Combine {group.name}, {group.side}. ', end='')
    files_number = len(group.files)
    layout = get_layout(files_number)
    target_image = create_target_image(group, layout)
    target_size = image_size(target_image)
    row_height = target_size.height // layout.rows
    column_width = target_size.width // layout.columns

    x_offset = y_offset = 0
    for i, file in enumerate(group.files):
        image = cv.imread(file.as_posix())
        if group.side.startswith('back') and with_rotation:
            try:
                image = rotate_text_up(image)
            except:
                print(file.as_posix())
                traceback.print_exc()
        put_image(image, target_image, Point(x_offset, y_offset))
        x_offset += column_width
        if (i + 1) % layout.columns == 0:
            x_offset = 0
            y_offset += row_height

    target_filename = target_folder / prepare_file_name(f'{group.name}_{group.side}{extension}')
    cv.imwrite(target_filename.as_posix(), target_image)


def get_layout(files_number: int) -> Layout:
    rows = int(files_number ** 0.5)
    columns = math.ceil(files_number / rows)
    return Layout(rows=rows, columns=columns)


def create_target_image(group: FilesGroup, layout: Layout) -> CvImage:
    sizes = [image_size(cv.imread(filename.as_posix()))
             for filename in group.files]
    max_width = max(size.width for size in sizes)
    max_height = max(size.height for size in sizes)

    target_height = layout.rows * max_height
    target_width = layout.columns * max_width

    print(f'Max width: {max_width}, max height: {max_height}; '
          f'images in group: {len(group.files)}; '
          f'layout: {layout.rows}x{layout.columns}; '
          f'target size: {target_width}, {target_height}')

    return np.zeros((target_height, target_width, 3))


def rotate_text_up(image: CvImage) -> CvImage:
    text_point = text_center_point(image)
    center = image_center(image)
    angle = angle_from_center_vertical(center.x, text_point)
    return rotate_image(image, angle)


def put_image(image: CvImage, target_image: CvImage, offset: Point) -> None:
    size = image_size(image)
    target_image[offset.y: offset.y + size.height, offset.x: offset.x + size.width, :] = image


def text_center_point(image: CvImage) -> Point:
    threshold = prepare_text_image(image)
    contour = text_contour(threshold)
    point = text_contour_center_point(contour)
    # cv.drawContours(image, [contour], -1, (255, 0, 0), 3)
    # cv.drawMarker(image, point, (255, 255, 0), thickness=40, markerSize=200)
    return point


def angle_from_center_vertical(center: int, point: Point) -> float:
    length_from_center = vector_length(center, center, point.x, point.y)
    angle = math.degrees(math.acos((center - point.y) / length_from_center))
    if point.x < center:
        angle = -angle
    return angle


def rotate_image(source: CvImage, angle: float) -> CvImage:
    rotation_matrix = cv.getRotationMatrix2D(image_center(source), angle, scale=1)
    return cv.warpAffine(source, rotation_matrix, dsize=image_size(source))


def prepare_text_image(image: CvImage) -> CvImage:
    threshold = convert_to_black_and_white(image, TEXT_HSV_RANGE)
    kernel = np.ones((3, 3), dtype=np.uint8)
    threshold = cv.morphologyEx(threshold, cv.MORPH_OPEN, kernel)
    return threshold


def text_contour(binary_image: CvImage) -> CvContour:
    def contour_on_plate(c: CvContour) -> bool:
        return all(vector_length(center.x, center.y, p[0][0], p[0][1]) < radius for p in c)

    RADIUS_GAP = 10
    center = image_center(binary_image)
    radius = image_size(binary_image).width // 2 - RADIUS_GAP
    _, contours, _ = cv.findContours(binary_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = (cv.convexHull(c) for c in contours)
    contours = [c for c in contours if cv.contourArea(c) >= 10 and contour_on_plate(c)]
    contour = cv.convexHull(np.vstack(contours))
    return contour


def text_contour_center_point(contour: CvContour) -> Point:
    moments = cv.moments(contour)
    x = int(moments["m10"] / moments["m00"])
    y = int(moments["m01"] / moments["m00"])
    return Point(x, y)


def vector_length(x0: float, y0: float, x1: float, y1: float) -> float:
    return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
