from collections import namedtuple

import numpy as np

CvImage = np.ndarray
CvContour = np.ndarray

FileDescription = namedtuple('ImageDescription', 'name number side')
StrainInfo = namedtuple('StrainInfo', 'name plates_number')
FilesGroup = namedtuple('FilesGroup', 'name side files')
ImageSize = namedtuple('ImageSize', 'width height')
Layout = namedtuple('Layout', 'rows columns')
ColorRange = namedtuple('Range', 'min max')
Point = namedtuple('Point', 'x y')
