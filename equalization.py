import cv2
import numpy as np
from matplotlib import pyplot as plt

from transform import transform


def histogram(channel):
    hist, _ = np.histogram(channel.ravel(), 256, [0, 256])
    return hist


def equalize(channel):
    hist = histogram(channel)
    func = []
    pre_sum = 0
    tot_sum = sum(hist)
    for num in hist:
        tmp = float(pre_sum+num) / tot_sum
        func.append(int(255*tmp+0.5))
        pre_sum += num

    out = np.zeros(channel.shape, np.uint8)
    for i in range(512):
        for j in range(512):
            out[i][j] = func[channel[i][j]]
    return out


if __name__ == '__main__':
    img = cv2.imread('img/lena.tiff')
    channels = cv2.split(img)
    color = ('b', 'g', 'r')
    for k, col in enumerate(color):
        hist_r = histogram(channels[k])
        plt.subplot(3, 3, k+1), plt.plot(hist_r, color=col), plt.title('{}'.format(col)), plt.xlim([0, 256])

    channels_ = np.zeros((3, 512, 512), dtype=np.uint8)
    for k, col in enumerate(color):
        channels_[k] = equalize(channels[k])
        hist_ = histogram(channels_[k])
        plt.subplot(3, 3, k+4), plt.plot(hist_, color=col), plt.title('{}'.format(col)), plt.xlim([0, 256])
    img_gbr_equal = channels_.transpose((1, 2, 0))

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    channels = cv2.split(img)
    img_yuv[:, :, 0] = equalize(channels[0])
    img_yuv_equal = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    plt.subplot(3, 3, 7), plt.imshow(transform(img)), plt.title('origin')
    plt.subplot(3, 3, 8), plt.imshow(transform(img_gbr_equal)), plt.title('gbr_equalization')
    plt.subplot(3, 3, 9), plt.imshow(transform(img_yuv_equal)), plt.title('yuv_equalization')
    plt.savefig('img/equalization/equalization.png')
    plt.show()

    cv2.imwrite('img/equalization/gbr_equalization.png', img_gbr_equal)
    cv2.imwrite('img/equalization/yuv_equalization.png', img_yuv_equal)
