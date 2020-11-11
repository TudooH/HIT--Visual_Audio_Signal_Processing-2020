import cv2
import numpy as np


def homomorphic_filter(src, d0=30, r1=0.8, rh=1.25, c=5):
    gray = src.copy()
    gray = np.float64(gray)
    rows, cols = gray.shape

    gray = np.log(gray + 1)
    gray_fft = np.fft.fft2(gray)
    gray_fft_shift = np.fft.fftshift(gray_fft)
    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2))
    D = np.sqrt(M ** 2 + N ** 2)
    Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + r1
    dst_fft_shift = Z * gray_fft_shift
    dst_shift = np.fft.ifftshift(dst_fft_shift)
    dst = np.fft.ifft2(dst_shift)
    dst = np.real(dst)
    dst = np.exp(dst) - 1
    dst = np.uint8(np.clip(dst, 0, 255))
    return dst


img = cv2.imread('../img/lena.tiff')
# img_ = homomorphic_filter(img)
img_ = np.zeros(img.shape, dtype=np.uint8)
for i in range(3):
    img_[:, :, i] = homomorphic_filter(img[:, :, i])

cv2.imshow('origin', img)
cv2.imshow('img', img_)
cv2.waitKey(0)
