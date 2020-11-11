import cv2
import numpy as np
import random
from matplotlib import pyplot as plt

from util.transform import transform


def save_and_show(origin, noise, noised, title):
    cv2.imwrite('../img/noise/{}_noise.png'.format(title), noised)
    plt.subplot(131), plt.imshow(transform(origin)), plt.title('origin')
    plt.subplot(132), plt.imshow(transform(noise)), plt.title('noise')
    plt.subplot(133), plt.imshow(transform(noised)), plt.title(title+'_noise')
    plt.savefig('../img/noise/{}_plt.png'.format(title))
    plt.show()


def sp_noise(img, pos):
    out = np.zeros(img.shape, np.uint8)
    noise = np.zeros(img.shape, np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rd = random.random()
            if rd < pos:
                out[i][j] = 0
                noise[i][j] = 0
            elif rd > 1-pos:
                out[i][j] = 255
                noise[i][j] = 255
            else:
                out[i][j] = img[i][j]
                noise[i][j] = 128
    return out, noise


def gauss_noise(img, mean=0, var=0.009):
    image = np.array(img / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    out = np.clip(out, 0, 1.0)
    out = np.uint8(out * 255)
    noise = np.uint8(noise * 255)
    return out, noise


if __name__ == '__main__':
    ori_img = cv2.imread('../img/lena.tiff')

    sp_img, noi_img = sp_noise(ori_img, 0.04)
    save_and_show(ori_img, noi_img, sp_img, 'sp')

    gauss_img, noi_img = gauss_noise(ori_img)
    save_and_show(ori_img, noi_img, gauss_img, 'gauss')
