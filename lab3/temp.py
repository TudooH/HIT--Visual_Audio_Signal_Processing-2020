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
    # img = cv2.resize(img, (600, 400))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)

    edged = cv2.Canny(gray, 30, 200)
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    contours = sorted(contours, key=cmp_to_key(lambda a, b: count_var(img, a)-count_var(img, b)))
    screenCnt = None

    for c in contours:
        # print(count_var(img, c))
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        if len(approx) == 4 and count_var(img, c) > 0.1:
            screenCnt = approx
            break

    if screenCnt is None:
        print("No contour detected")
        return
    else:
        detected = 1

    # print(count_var(img, screenCnt))
    if detected == 1:
        cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [screenCnt], 0, 255, -1)
    # new_image = cv2.bitwise_and(img, img, mask=mask)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    # return topy / img.shape[1], topx / img.shape[0], bottomy / img.shape[1], bottomx / img.shape[0]

    Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

    text = pytesseract.image_to_string(Cropped, config='--psm 11')
    print("programming_fever's License Plate Recognition\n")
    print("Detected license plate Number is:", text)
    img = cv2.resize(img, (500, 300))
    Cropped = cv2.resize(Cropped, (400, 200))
    cv2.imshow('car', img)
    cv2.imshow('Cropped', Cropped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # img = cv2.imread('../img/test.jpeg', cv2.IMREAD_COLOR)
    detection('../data/004.jpeg')
