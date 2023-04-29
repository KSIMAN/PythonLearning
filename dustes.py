import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загружаем изображение
img = cv2.imread('photos/2.jpg')

# Преобразуем изображение в оттенки серого
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Применяем адаптивный порог для выделения пылинок
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)

median = cv2.medianBlur(thresh, 5)
median = cv2.GaussianBlur(median, (3, 3), 0) #размыть
# Находим контуры на изображении
contours, hierarchy = cv2.findContours(median,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# Инициализируем счетчики для разных размеров пылинок
small_count = 0
medium_count = 0
large_count = 0

# Проходим по всем контурам и определяем их размер
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 300:
        continue
    elif area < 500:
        medium_count += 1
    else:
        large_count += 1

# Выводим результаты
print("Количество мелких частиц:", small_count)
print("Количество средних частиц:", medium_count)
print("Количество крупных частиц:", large_count)

# Визуализируем результаты
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Исходное изображение")
plt.show()

plt.imshow(thresh, cmap='gray')
plt.title("Выделенные крошки")
plt.show()