from functools import cmp_to_key
import cv2
import imutils
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'


def contours_var(gray, roi):
    if cv2.contourArea(roi) == 0:
        return 0

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

    edged = cv2.Canny(gray, 30, 180)
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    contours = sorted(contours, key=cmp_to_key(lambda a, b: contours_var(gray, a) - contours_var(gray, b)))
    screenCnt = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        if len(approx) == 4 and contours_var(gray, approx) > 0:
            screenCnt = approx
            break

    if screenCnt is None:
        return 0, 0, 0, 0, None, None

    cv2.drawContours(img, [screenCnt], -1, (255, 0, 0), 3)

    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [screenCnt], 0, 255, -1)

    x, y = np.where(mask == 255)
    top_x, top_y = np.min(x), np.min(y)
    bottom_x, bottom_y = np.max(x), np.max(y)
    cropped = gray[top_x: bottom_x + 1, top_y: bottom_y + 1]
    return top_y / img.shape[1], top_x / img.shape[0], bottom_y / img.shape[1], bottom_x / img.shape[0], img, cropped


def image2txt(cropped):
    return pytesseract.image_to_string(cropped, config='--psm 7')


if __name__ == '__main__':
    _, _, _, _, image, area = detection('../data/000.jpeg')
    txt = image2txt(area)
    print(txt)
    cv2.imshow('image', image)
    cv2.imshow('area', area)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
