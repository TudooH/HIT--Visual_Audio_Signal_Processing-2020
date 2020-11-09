import cv2
import numpy as np


def homomorphic_filter(src, d0=10, r1=0.5, rh=2, c=4, h=2.0, l=0.5):
    gray = src.copy()
    if len(src.shape) > 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)
    rows, cols = gray.shape

    gray_fft = np.fft.fft2(gray)
    gray_fft_shift = np.fft.fftshift(gray_fft)
    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2))
    D = np.sqrt(M ** 2 + N ** 2)
    Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + r1
    dst_fft_shift = Z * gray_fft_shift
    dst_fft_shift = (h - l) * dst_fft_shift + l
    dst_shift = np.fft.ifftshift(dst_fft_shift)
    dst = np.fft.ifft2(dst_shift)
    dst = np.real(dst)
    dst = np.uint8(np.clip(dst, 0, 255))
    return dst


img = cv2.imread('img/lena.tiff')
for i in range(3):
    img[:, :, i] = homomorphic_filter(img[:, :, i])
cv2.imshow('img', img)
cv2.waitKey(0)
