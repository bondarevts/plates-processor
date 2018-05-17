import cv2 as cv
import numpy as np

from commands.types import ColorRange
from commands.types import CvImage
from commands.types import ImageSize
from commands.types import Point


def convert_to_black_and_white(img: CvImage, hsv_range: ColorRange) -> CvImage:
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    return cv.inRange(hsv, hsv_range.min, hsv_range.max)


def image_size(image: CvImage) -> ImageSize:
    height, width, *_ = image.shape
    return ImageSize(width=width, height=height)


def image_center(image: CvImage) -> Point:
    size = image_size(image)
    return Point(size.height // 2, size.width // 2)


def clear_outside_circle(image: CvImage, center: Point, radius: int) -> CvImage:
    mask = np.zeros_like(image)
    cv.circle(mask, center, radius, color=(255, 255, 255), thickness=-1)
    result = image.copy()
    result[mask != 255] = 0
    return result


def put_image(image: CvImage, target_image: CvImage, offset: Point) -> None:
    size = image_size(image)
    target_image[offset.y: offset.y + size.height, offset.x: offset.x + size.width, :] = image


def rotate_image(source: CvImage, angle: float) -> CvImage:
    rotation_matrix = cv.getRotationMatrix2D(image_center(source), angle, scale=1)
    return cv.warpAffine(source, rotation_matrix, dsize=image_size(source))
