import cv2.cv2 as cv2
import numpy as np

flower = cv2.imread('../../Sample Images/flowerSalt.jpg')
flower = cv2.resize(flower, (800, 600))

# filter = np.array([(1, 1, 1), (1, 1, 1), (1, 1, 1)]) * (1 / 9)  # filter 3x3
filter = np.array([(1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1)]) * (1/25)

img_shape = flower.shape
filter_shape = filter.shape

flower_gray = cv2.cvtColor(flower, cv2.COLOR_BGR2GRAY)
cv2.imshow("Before filter", flower_gray)

# add padding to maintain size after filtering
row = img_shape[0] + filter_shape[0] - 1
col = img_shape[1] + filter_shape[1] - 1
new_img_arr = np.zeros((row, col))

for i in range(img_shape[0]):
    for j in range(img_shape[1]):
        new_img_arr[i + int((filter_shape[0] - 1) / 2), j + int((filter_shape[1] - 1) / 2)] = flower_gray[i, j]

print(new_img_arr)

for i in range(img_shape[0]):
    for j in range(img_shape[1]):
        temp = new_img_arr[i:i + filter_shape[0], j:j + filter_shape[1]]
        res = np.prod(temp ** filter)
        flower_gray[i, j] = res

cv2.imshow("After filter", flower_gray)
cv2.waitKey(0)
