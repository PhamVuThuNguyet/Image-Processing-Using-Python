import cv2.cv2 as cv2
import numpy as np
import math


def padding(img):
    filter_shape = 9

    row = img.shape[0] + filter_shape - 1
    col = img.shape[1] + filter_shape - 1
    new_img_arr = np.zeros((row, col))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img_arr[i + int((filter_shape - 1) / 2), j + int((filter_shape - 1) / 2)] = img[i, j]

    return new_img_arr


def log_kernel(size, sigma):
    log_kernel = np.zeros((size, size), np.float32)
    x = (size - 1) // 2
    for i in range(-x, x + 1):
        for j in range(-x, x + 1):
            norm = math.pow(i, 2) + math.pow(j, 2)
            log_kernel[i + x, j + x] = (-1 / (math.pi * math.pow(sigma, 4))) * \
                                       (1 - (norm / (2 * math.pow(sigma, 2)))) * \
                                       math.exp(-norm / (2 * math.pow(sigma, 2)))
    return log_kernel


def LoG(img, new_img_arr, filter):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            temp = new_img_arr[i:i + filter.shape[0], j:j + filter.shape[1]]
            res = np.sum(temp * filter)
            img[i, j] = res

    return img.astype(np.uint8)


if __name__ == '__main__':
    flower = cv2.imread('../Sample Images/img_9.png')
    flower = cv2.resize(flower, (800, 600))

    img_shape = flower.shape

    flower_gray = cv2.cvtColor(flower, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Before filter", flower_gray)

    new_img_arr = padding(flower_gray)
    kernel = log_kernel(9, 1.4)
    new_img_arr = LoG(flower_gray, new_img_arr, kernel)
    print(new_img_arr)
    cv2.imshow("After filter", new_img_arr)
    cv2.waitKey(0)
