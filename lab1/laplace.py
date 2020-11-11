import cv2
import numpy as np
from matplotlib import pyplot as plt


def laplace(img):
    lap = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    out = np.zeros(img.shape, np.uint8)
    for i in range(img.shape[0]-2):
        for j in range(img.shape[1]-2):
            pixels = img[i:i+3, j:j+3]
            tmp = np.sum(pixels*lap)
            if tmp < 0:
                tmp = 0
            if tmp > 255:
                tmp = 255
            out[i][j] = tmp
    return out


if __name__ == '__main__':
    image = cv2.imread('../img/lena.tiff', 0)

    my_laplace = laplace(image)
    plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('origin')
    plt.subplot(122), plt.imshow(my_laplace, cmap='gray'), plt.title('laplace')
    plt.show()

    cv2.imwrite('../img/laplace/my_laplace.png', my_laplace)
