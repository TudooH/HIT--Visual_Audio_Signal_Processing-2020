from functools import cmp_to_key
import cv2
import imutils
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'


def cal_points(gray, roi):
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [roi], 0, 255, -1)

    x, y = np.where(mask == 255)
    top_x, top_y = np.min(x), np.min(y)
    bottom_x, bottom_y = np.max(x), np.max(y)

    return top_x, top_y, bottom_x, bottom_y


def cal_ratio(item):
    return abs(3.2 - (item[1][3] - item[1][1]) / (item[1][2] - item[1][0]))


def image2txt(cropped):
    return pytesseract.image_to_string(cropped, config='--psm 7')


def search(filename):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    # if img.shape[1] > 800:
    #     img = cv2.resize(img, (int(img.shape[1] * 800 / img.shape[0]), 800))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    two = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 10)

    ex = cv2.bitwise_not(two)
    kernel = np.ones((5, 0), dtype=np.uint8)
    ex = cv2.morphologyEx(ex, cv2.MORPH_CLOSE, kernel, iterations=4)

    edged = cv2.Canny(ex, 30, 180)

    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    points = []
    for i, contour in enumerate(contours):
        points.append([i, cal_points(gray, contour)])

    points = sorted(points, key=cal_ratio)

    top_x, top_y, bottom_x, bottom_y = points[0][1]
    cropped = gray[top_x: bottom_x + 1, top_y: bottom_y + 1]
    cv2.rectangle(img, (top_y, top_x+1), (bottom_y, bottom_x+1), (255, 0, 0), 2)
    if cropped.shape[1] < 70:
        cropped = cv2.resize(cropped, (int(cropped.shape[1] * 70 / cropped.shape[0]), 70))
    if img.shape[1] > 800:
        img = cv2.resize(img, (int(img.shape[1] * 800 / img.shape[0]), 800))
    return top_y / img.shape[1], top_x / img.shape[0], bottom_y / img.shape[1], bottom_x / img.shape[0], img, cropped


if __name__ == '__main__':
    _, _, _, _, image, area = search('../img/0.jpg')
    txt = image2txt(area)
    print('txt: {}'.format(txt))
    cv2.imshow('image', image)
    cv2.imshow('area', area)
    cv2.imwrite('area.jpg', area)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 34, 89
