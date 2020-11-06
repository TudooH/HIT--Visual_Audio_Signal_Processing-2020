import cv2
import numpy as np
from matplotlib import pyplot as plt


def histogram(channel):
    hist, _ = np.histogram(channel.ravel(), 256, [0, 256])
    return hist


def mapping(channel):
    hist = histogram(channel)
    func = []
    pre_sum = 0
    tot_sum = sum(hist)
    for num in hist:
        tmp = float(pre_sum + num) / tot_sum
        func.append(int(255 * tmp + 0.5))
        pre_sum += num
    return func


def equalize(channel, standard=None):
    func = mapping(channel)
    if standard is None:
        st = [i for i in range(256)]
    else:
        st = mapping(standard)

    i = 0
    j = 0
    while i < 256:
        if func[i] == st[j]:
            func[i] = j
            i += 1
        elif func[i] < st[j]:
            func[i] = j-1
            i += 1
        else:
            j += 1

        if j == 256:
            while i < 256:
                func[i] = 255
                i += 1

    out = np.zeros(channel.shape, np.uint8)
    for i in range(512):
        for j in range(512):
            out[i][j] = func[channel[i][j]]
    return out


if __name__ == '__main__':
    img = cv2.imread('img/lena.tiff')
    standard_img = cv2.imread('img/equalization/standard2.jpg')

    color = ('b', 'g', 'r')

    channels = cv2.split(img)
    channels_ = np.zeros((3, 512, 512), dtype=np.uint8)
    for k, col in enumerate(color):
        hist_r = histogram(channels[k])
        plt.subplot(4, 3, k + 1), plt.plot(hist_r, color=col), plt.title('origin_{}'.format(col)), plt.xlim([0, 256])
    for k, col in enumerate(color):
        channels_[k] = equalize(channels[k])
        hist_ = histogram(channels_[k])
        plt.subplot(4, 3, k + 4), plt.plot(hist_, color=col), plt.title('equalized_{}'.format(col)), plt.xlim([0, 256])
    img_gbr_equal = channels_.transpose((1, 2, 0))

    channels_ = np.zeros((3, 512, 512), dtype=np.uint8)
    channels_standard = cv2.split(standard_img)
    for k, col in enumerate(color):
        hist_r = histogram(channels_standard[k])
        plt.subplot(4, 3, k + 7), plt.plot(hist_r, color=col), plt.title('standard_{}'.format(col)), plt.xlim([0, 256])

    for k, col in enumerate(color):
        channels_[k] = equalize(channels[k], channels_standard[k])
        hist_ = histogram(channels_[k])
        plt.subplot(4, 3, k + 10), plt.plot(hist_, color=col), plt.title('equalized_{}'.format(col)), plt.xlim([0, 256])
    img_standard_equal = channels_.transpose((1, 2, 0))
    plt.savefig('img/equalization/equalization.png')
    plt.show()

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    standard_yuv = cv2.cvtColor(standard_img, cv2.COLOR_BGR2YUV)
    channels = cv2.split(img)
    channels_standard = cv2.split(standard_yuv)
    img_yuv[:, :, 0] = equalize(channels[0])
    img_yuv_equal = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    img_yuv[:, :, 0] = equalize(channels[0], channels_standard[0])
    img_yuv_standard = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    cv2.imwrite('img/equalization/gbr_equalization.png', img_gbr_equal)
    cv2.imwrite('img/equalization/yuv_equalization.png', img_yuv_equal)
    cv2.imwrite('img/equalization/gbr_standard.png', img_standard_equal)
    cv2.imwrite('img/equalization/yuv_standard.png', img_yuv_standard)

    cv2.imshow('gbr', img_gbr_equal)
    cv2.imshow('gbr_standard', img_standard_equal)
    cv2.imshow('yuv', img_yuv_equal)
    cv2.imshow('yuv_standard', img_yuv_standard)
    cv2.waitKey(0)
