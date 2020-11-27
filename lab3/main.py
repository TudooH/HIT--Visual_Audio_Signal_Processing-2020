import cv2

from lab3.search import search
from lab3.detection import detection


def processing(filename):
    _, _, _, _, image, area = search(filename)
    print('Car number: {}'.format(detection(area)))

    cv2.imshow('car', image)
    cv2.imshow('area', area)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    processing('../img/cars/0.jpg')
