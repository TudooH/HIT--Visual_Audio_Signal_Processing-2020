import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


def gauss(img):
    sigma1 = sigma2 = 1
    gaussian = np.array([[math.exp(-1 / 2 * (np.square(i-3)/np.square(sigma1) + (np.square(j-3)/np.square(sigma2))))
                          / (2 * math.pi * 1 * 1) for j in range(5)] for i in range(5)])
    gaussian = gaussian / np.sum(gaussian)
    out = np.array([[np.sum(img[i: i+5, j: j+5]*gaussian) for j in range(img.shape[0]-5)]
                   for i in range(img.shape[1]-5)], dtype=float)
    return out


def gradient(img):
    w = img.shape[0]-1
    h = img.shape[1]-1
    dx = np.zeros([w, h])
    dy = np.zeros([w, h])
    d = np.zeros([w, h])
    for i in range(w):
        for j in range(h):
            dx[i, j] = img[i, j+1] - img[i, j]
            dy[i, j] = img[i+1, j] - img[i, j]
            d[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))

    nms = np.copy(d)
    nms[0, :] = nms[w-1, :] = nms[:, 0] = nms[:, h-1] = 0
    for i in range(1, w-1):
        for j in range(1, h-1):
            if d[i, j] == 0:
                nms[i, j] = 0
            else:
                grad_x = dx[i, j]
                grad_y = dy[i, j]
                grad_temp = d[i, j]

                if np.abs(grad_y) > np.abs(grad_x):
                    weight = np.abs(grad_x) / np.abs(grad_y)
                    grad2 = d[i-1, j]
                    grad4 = d[i+1, j]
                    if grad_x * grad_y > 0:
                        grad1 = d[i-1, j-1]
                        grad3 = d[i+1, j+1]
                    else:
                        grad1 = d[i-1, j+1]
                        grad3 = d[i+1, j-1]
                else:
                    weight = np.abs(grad_y) / np.abs(grad_x)
                    grad2 = d[i, j-1]
                    grad4 = d[i, j+1]
                    if grad_x * grad_y > 0:
                        grad1 = d[i+1, j-1]
                        grad3 = d[i-1, j+1]
                    else:
                        grad1 = d[i-1, j-1]
                        grad3 = d[i+1, j+1]

                grad_temp1 = weight*grad1 + (1-weight)*grad2
                grad_temp2 = weight*grad3 + (1-weight)*grad4
                if grad_temp >= grad_temp1 and grad_temp >= grad_temp2:
                    nms[i, j] = grad_temp
                else:
                    nms[i, j] = 0

    out = np.zeros([w, h])
    low = 0.08 * np.max(nms)
    high = 0.16 * np.max(nms)
    for i in range(1, w-1):
        for j in range(1, h-1):
            if nms[i, j] < low:
                out[i, j] = 0
            elif nms[i, j] > high:
                out[i, j] = 255
            elif ((nms[i-1, j-1: j+1] > high).any() or (nms[i+1, j-1: j+1] > high).any()
                  or (nms[i, [j-1, j+1]] > high).any()):
                out[i, j] = 255
    return out


def canny(img):
    img = gauss(img)
    return gradient(img)


if __name__ == '__main__':
    gray = cv2.imread('../img/lena.tiff', 0)
    my_canny = canny(gray)
    cv2_canny = cv2.Canny(gray, 80, 150)
    cv2.imwrite('../img/canny/my_canny.png', my_canny)
    cv2.imwrite('../img/canny/cv2_canny.png', cv2_canny)
    plt.subplot(131), plt.imshow(gray, cmap='gray'), plt.title('gray')
    plt.subplot(132), plt.imshow(my_canny, cmap='gray'), plt.title('my_canny')
    plt.subplot(133), plt.imshow(cv2_canny, cmap='gray'), plt.title('cv2_canny')
    plt.savefig('../img/canny/canny.png')
    plt.show()
