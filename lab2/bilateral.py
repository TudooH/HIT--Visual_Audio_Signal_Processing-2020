import numpy as np
import cv2
import math


def bilateral_filter(channel, size, sigma_d, sigma_r):
    mid = int(size / 2)
    w_d = np.array([[math.exp(-((i - mid) ** 2 + (j - mid) ** 2) / (2 * sigma_d ** 2))
                     for j in range(size)] for i in range(size)])

    channel = channel.astype(np.float)
    out = np.zeros((channel.shape[0] - size + 1, channel.shape[1] - size + 1), dtype=np.uint8)
    for i in range(channel.shape[0] - size + 1):
        for j in range(channel.shape[1] - size + 1):
            w_r = np.array([[math.exp(-(channel[i + ii][j + jj] - channel[i + mid][j + mid]) ** 2 / (2 * sigma_r ** 2))
                            for jj in range(size)] for ii in range(size)])
            w = w_d * w_r
            w = w / sum(sum(w))
            out[i][j] = sum(sum(w * channel[i: i+size, j: j+size]))
    return out


if __name__ == '__main__':
    img = cv2.imread('../img/lena.tiff')
    channels = cv2.split(img)

    Size = 5
    img_ = np.zeros((img.shape[0] - Size + 1, img.shape[1] - Size + 1, 3), dtype=np.uint8)
    for c in range(3):
        print('start channel {}'.format(c))
        img_[:, :, c] = bilateral_filter(channels[c], Size, 5, 30)
        print('finished')

    cv2.imwrite('../img/bilateral/bilateral.png', img_)
    cv2.imshow('origin', img)
    cv2.imshow('img', img_)
    cv2.waitKey(0)
