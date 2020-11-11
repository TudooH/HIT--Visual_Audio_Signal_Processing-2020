import cv2
import numpy as np


def homomorphic_filter(gray, d0=30, r1=0.75, rh=1.75, c=5):
    gray = np.float64(gray)
    rows, cols = gray.shape

    gray = np.log(gray + 1)
    gray_fft = np.fft.fft2(gray)

    for i in range(rows):
        for j in range(cols):
            tmp = (i - rows // 2) ** 2 + (j - cols // 2) ** 2
            gray_fft[i][j] = ((rh - r1) * (1 - np.exp(-c * (tmp ** 2 / d0 ** 2))) + r1) * gray_fft[i][j]

    out = np.fft.ifft2(gray_fft)
    out = np.real(out)
    out = np.exp(out) - 1
    out = np.uint8(np.clip(out, 0, 255))
    return out


if __name__ == '__main__':
    img = cv2.imread('../img/homomorphic/test.png')
    img_ = np.zeros(img.shape, dtype=np.uint8)
    for cc in range(3):
        img_[:, :, cc] = homomorphic_filter(img[:, :, cc])

    cv2.imshow('origin', img)
    cv2.imshow('img', img_)
    cv2.imwrite('../img/homomorphic/homomorphic.png', img_)
    cv2.waitKey(0)
