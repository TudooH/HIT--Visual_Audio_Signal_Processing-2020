from functools import cmp_to_key

import cv2
import imutils
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'


def count_var(img, roi):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [roi], 0, 255, -1)

    x, y = np.where(mask == 255)
    topx, topy = np.min(x), np.min(y)
    bottomx, bottomy = np.max(x), np.max(y)

    _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return np.var(gray[topx:bottomx + 1, topy:bottomy + 1]) / cv2.contourArea(roi)


def detection(filename):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # cv2.imshow('gray', gray)

    edged = cv2.Canny(gray, 30, 200)
    # cv2.imshow('canny', edged)
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    contours = sorted(contours, key=cmp_to_key(lambda x, y: count_var(img, x)-count_var(img, y)))
    for i, contour in enumerate(contours):
        print(i, count_var(img, contour), len(cv2.approxPolyDP(contour, 0.018 * cv2.arcLength(contour, True), True)))

    cv2.drawContours(img, contours, 4, (0, 255, 0), 1)

    cv2.imshow('car', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detection('../data/008.jpeg')
