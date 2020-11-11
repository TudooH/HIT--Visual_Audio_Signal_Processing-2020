import cv2
import numpy as np
from matplotlib import pyplot as plt


def dft(img):
    img = img.astype(np.float32)
    h, w, c = img.shape

    out = np.zeros((h, w, c), dtype=np.complex)
    x = np.tile(np.arange(w), (h, 1))
    y = np.arange(h).repeat(w).reshape(h, -1)

    for c in range(c):
        for v in range(h):
            for u in range(w):
                out[v, u, c] = np.sum(img[:, :, c] * np.exp(-2j * np.pi * (x * u / w + y * v / h))) / np.sqrt(h * w)

    return out


def i_dft(img_complex):
    h, w, c = img_complex.shape

    out = np.zeros((h, w, c), dtype=np.float32)
    x = np.tile(np.arange(w), (h, 1))
    y = np.arange(h).repeat(w).reshape(h, -1)

    for c in range(c):
        for v in range(h):
            for u in range(w):
                out[v, u, c] = np.abs(np.sum(img_complex[:, :, c] * np.exp(2j * np.pi * (x * u / w + y * v / h)))) / np.sqrt(w * h)

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)
    return out


def dft_shift(channel):
    img_shift = 1j + np.zeros(channel.shape)
    h = channel.shape[0]
    w = channel.shape[1]
    img_shift[h >> 1: h, w >> 1:w] = channel[0: h >> 1, 0:w >> 1]
    img_shift[h >> 1: h, 0: w >> 1] = channel[0: h >> 1, w >> 1: w]
    img_shift[0: h >> 1, w >> 1: w] = channel[h >> 1: h, 0: w >> 1]
    img_shift[0: h >> 1, 0: w >> 1] = channel[h >> 1: h, w >> 1: w]
    return img_shift


def dft_map(shift):
    mapping = np.log2(np.abs(shift))
    mapping = np.round((mapping - np.min(mapping)) / (np.max(mapping) - np.min(mapping)) * 255)
    mapping = mapping.astype('uint8')
    return mapping


image = cv2.imread("../img/homomorphic/test.jpg")
dft_img = dft(image)
i_dft_img = i_dft(dft_img)

map_img0 = dft_map(dft_shift(dft_img[:, :, 0]))
map_img1 = dft_map(dft_shift(dft_img[:, :, 1]))
map_img2 = dft_map(dft_shift(dft_img[:, :, 2]))

plt.subplot(131), plt.imshow(map_img0, cmap='gray'), plt.title('channel0')
plt.subplot(132), plt.imshow(map_img1, cmap='gray'), plt.title('channel1')
plt.subplot(133), plt.imshow(map_img2, cmap='gray'), plt.title('channel2')
plt.savefig('../img/homomorphic/dft_map.png')
plt.show()

cv2.imshow('result', i_dft_img)
cv2.imwrite('../img/homomorphic/result.png', i_dft_img)
cv2.waitKey(0)
