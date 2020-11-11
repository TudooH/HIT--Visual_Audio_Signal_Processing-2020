import numpy as np


def transform(img):
    tmp = np.zeros(img.shape, np.uint8)
    tmp[:, :, 0] = img[:, :, 2]
    tmp[:, :, 1] = img[:, :, 1]
    tmp[:, :, 2] = img[:, :, 0]
    return tmp
