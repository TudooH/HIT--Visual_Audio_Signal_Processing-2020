import cv2
import numpy as np
from matplotlib import pyplot as plt


def sobel(img):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    out = np.zeros(img.shape, np.uint8)
    out_x = np.zeros(img.shape, np.uint8)
    out_y = np.zeros(img.shape, np.uint8)
    for i in range(img.shape[0]-2):
        for j in range(img.shape[1]-2):
            pixels = img[i:i+3, j:j+3]
            x = np.sum(pixels*sobel_x)
            y = np.sum(pixels*sobel_y)
            out_x[i][j] = abs(x)
            out_y[i][j] = abs(y)
            out[i][j] = (x**2 + y**2)**0.5
    return out, out_x, out_y


if __name__ == '__main__':
    image = cv2.imread('img/lena.tiff', 0)

    my_sobel, my_sobel_x, my_sobel_y = sobel(image)
    plt.subplot(231), plt.imshow(my_sobel, cmap='gray'), plt.title('my_sobel')
    plt.subplot(232), plt.imshow(my_sobel_x, cmap='gray'), plt.title('x')
    plt.subplot(233), plt.imshow(my_sobel_y, cmap='gray'), plt.title('y')

    cv2_sobel_x = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    cv2_sobel_y = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    cv2_sobel_x = cv2.convertScaleAbs(cv2_sobel_x)
    cv2_sobel_y = cv2.convertScaleAbs(cv2_sobel_y)
    cv2_sobel = cv2.addWeighted(cv2_sobel_x, 0.5, cv2_sobel_y, 0.5, 0)
    plt.subplot(234), plt.imshow(cv2_sobel, cmap='gray'), plt.title('cv2_sobel')
    plt.subplot(235), plt.imshow(cv2_sobel_x, cmap='gray'), plt.title('x')
    plt.subplot(236), plt.imshow(cv2_sobel_y, cmap='gray'), plt.title('y')

    plt.show()

    cv2.imwrite('img/sobel/my_sobel.png', my_sobel)
    cv2.imwrite('img/sobel/my_sobel_x.png', my_sobel_x)
    cv2.imwrite('img/sobel/my_sobel_y.png', my_sobel_y)
    cv2.imwrite('img/sobel/cv2_sobel.png', cv2_sobel)
    cv2.imwrite('img/sobel/cv2_sobel_x.png', cv2_sobel_x)
    cv2.imwrite('img/sobel/cv2_sobel_y.png', cv2_sobel_y)
