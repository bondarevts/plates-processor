import cv2 as cv

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
