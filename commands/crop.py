from pathlib import Path
from typing import Iterable

import cv2 as cv
import numpy as np

from commands.files_utils import files_from
from commands.files_utils import prepare_file_name
from commands.image_utils import convert_to_black_and_white
from commands.types import ColorRange
from commands.types import CvContour
from commands.types import CvImage

PLATE_HSV_RANGE = ColorRange(
    min=np.array((0, 0, 75), np.uint8),
    max=np.array((255, 255, 214), np.uint8)
)


def crop_files(extension: str, input_folder: Path, target_folder: Path) -> None:
    files = files_from(input_folder, extension)
    for file in files:
        print(f'Crop {file}', end='; ')
        image = extract_plate(file)

        target_path = target_folder / prepare_file_name(f'{file.stem}{file.suffix}')
        print(f'saved in {target_path}')
        cv.imwrite(target_path.as_posix(), image)


def extract_plate(image_path: Path) -> CvImage:
    image = cv.imread(image_path.as_posix())
    contours = get_complex_contours(image)
    plate_contour = get_plate_contour(contours)
    return crop_plate_square(image, plate_contour)


def get_complex_contours(image: CvImage) -> Iterable[CvContour]:
    threshold = convert_to_black_and_white(image, PLATE_HSV_RANGE)

    _, contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    hull_contours = map(cv.convexHull, contours)
    complex_shape_vertices_number = 100
    return (c for c in hull_contours if len(c) > complex_shape_vertices_number)


def get_plate_contour(contours: Iterable[CvContour]) -> CvContour:
    *_, plate_contour = max((cv.contourArea(contour), i, contour)
                            for i, contour in enumerate(contours))
    return plate_contour


def crop_plate_square(image: CvImage, plate_contour: CvContour) -> CvImage:
    fill_black_outside_contour(image, plate_contour)
    x, y, w, h = cv.boundingRect(plate_contour)
    size = max(w, h)
    return image[y:y + size, x:x + size]


def fill_black_outside_contour(image: CvImage, plate_contour: CvContour) -> None:
    mask = np.zeros_like(image)
    (x, y), radius = cv.minEnclosingCircle(plate_contour)
    center = (int(x), int(y))
    radius = int(radius)
    cv.circle(mask, center, radius, color=(255, 255, 255), thickness=-1)
    image[mask != 255] = 0
