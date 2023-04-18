import cv2
from matplotlib import pyplot as plt
import numpy as np


def conturs1(img, save):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    low_color = (0, 0, 0)
    high_color = (110, 120, 140)
    only_detail = cv2.inRange(image, low_color, high_color)

    moments = cv2.moments(only_detail, 1)
    x_moment = moments['m01']
    y_moment = moments['m10']
    area = moments['m00']
    x = int(x_moment / area)
    y = int(y_moment / area)
    cv2.putText(only_detail, "Detail", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    edge = cv2.Canny(image, 50, 260)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    plt.figure(figsize=(15, 15))
    cv2.imshow(img, edge)
    cv2.imwrite(save, edge)

    cv2.waitKey()
    cv2.destroyAllWindows()


def conturs2(img, save):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    low_color = (0, 0, 0)
    high_color = (250, 110, 250)
    only_detail = cv2.inRange(image, low_color, high_color)

    plt.figure(figsize=(15, 15))
    cv2.imshow(img, only_detail)
    cv2.imwrite(save, only_detail)

    cv2.waitKey()
    cv2.destroyAllWindows()


def something(img, save):
    image = cv2.imread(img)

    hsv_min = np.array((0, 0, 0), np.uint8)
    hsv_max = np.array((150, 200, 150), np.uint8)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # меняем цветовую модель с BGR на HSV
    thresh = cv2.inRange(hsv, hsv_min, hsv_max)  # применяем цветовой фильтр
    # ищем контуры и складируем их в переменную contours
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # отображаем контуры поверх изображения
    cv2.drawContours(image, contours, -1, (255, 0, 0), 2, cv2.LINE_AA, hierarchy, 1)
    cv2.imwrite(save, image)
    cv2.imshow('contours', image)  # выводим итоговое изображение в окно

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    something('kontur_detali.jpg', "kontur_detali_save_some.jpg")
