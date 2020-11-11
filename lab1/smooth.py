import cv2
import numpy as np
from matplotlib import pyplot as plt

from util.transform import transform


def smooth(img, pattern, r):
    out = np.zeros(img.shape, np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x1 = max(0, i-int(r/2))
            x2 = min(img.shape[0]-1, i+int(r/2))
            y1 = max(0, j-int(r/2))
            y2 = min(img.shape[1]-1, j+int(r/2))
            pixels = img[x1:x2+1, y1:y2+1, :]
            pixels = pixels.reshape(pixels.shape[0]*pixels.shape[1], pixels.shape[2])
            if pattern == 'mean':
                pixel = np.mean(pixels, axis=0)
            else:
                pixel = np.median(pixels, axis=0)
            out[i][j] = pixel
    return out


if __name__ == '__main__':
    gauss_noise = cv2.imread('../img/noise/gauss_noise.png')
    gauss_mean_smooth = smooth(gauss_noise, 'mean', 3)
    gauss_median_smooth = smooth(gauss_noise, 'median', 3)
    sp_noise = cv2.imread('../img/noise/sp_noise.png')
    sp_mean_smooth = smooth(sp_noise, 'mean', 3)
    sp_median_smooth = smooth(sp_noise, 'median', 3)

    cv2.imwrite('../img/smooth/gauss_mean_smooth.png', gauss_mean_smooth)
    cv2.imwrite('../img/smooth/gauss_median_smooth.png', gauss_median_smooth)
    cv2.imwrite('../img/smooth/sp_mean_smooth.png', sp_mean_smooth)
    cv2.imwrite('../img/smooth/sp_median_smooth.png', sp_median_smooth)

    plt.subplot(231), plt.imshow(transform(gauss_noise)), plt.title('gauss_noise')
    plt.subplot(232), plt.imshow(transform(gauss_mean_smooth)), plt.title('gauss_mean_smooth')
    plt.subplot(233), plt.imshow(transform(gauss_median_smooth)), plt.title('gauss_median_smooth')
    plt.subplot(234), plt.imshow(transform(sp_noise)), plt.title('sp_noise')
    plt.subplot(235), plt.imshow(transform(sp_mean_smooth)), plt.title('sp_mean_smooth')
    plt.subplot(236), plt.imshow(transform(sp_median_smooth)), plt.title('sp_median_smooth')
    plt.savefig('../img/smooth/smooth.png')
    plt.show()
