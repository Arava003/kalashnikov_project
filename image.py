import cv2
from matplotlib import pyplot as plt
import numpy as np

image = cv2.imread("kontur_detali.jpg")
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

t_lower = 50
t_upper = 150

# Applying the Canny Edge filter
# edge = cv2.Canny(image, t_lower, t_upper)

plt.figure(figsize=(15, 15))
# cv2.imshow('edge1', edge)
# cv2.imwrite('edge1.jpg', edge)

# plt.show()


hsv_min = np.array((50, 50, 50), np.uint8)
hsv_max = np.array((150, 150, 150), np.uint8)

if __name__ == '__main__':
    print(__doc__)

    fn = 'kontur_detali.jpg'  # путь к файлу с картинкой
    img = cv2.imread(fn)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # меняем цветовую модель с BGR на HSV
    thresh = cv2.inRange(hsv, hsv_min, hsv_max)  # применяем цветовой фильтр
    # ищем контуры и складируем их в переменную contours
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # отображаем контуры поверх изображения
    cv2.drawContours(img, contours, -1, (255, 0, 0), 3, cv2.LINE_AA, hierarchy, 1)
    cv2.imshow('contours', img)  # выводим итоговое изображение в окно

    cv2.waitKey()
    cv2.destroyAllWindows()
