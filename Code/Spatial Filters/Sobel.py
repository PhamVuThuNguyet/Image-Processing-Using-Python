import cv2.cv2 as cv2
import numpy as np
import math

def padding(img):
    filter_shape = 3

    row = img.shape[0] + filter_shape - 1
    col = img.shape[1] + filter_shape - 1
    new_img_arr = np.zeros((row, col))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img_arr[i + int((filter_shape - 1) / 2), j + int((filter_shape - 1) / 2)] = img[i, j]

    return new_img_arr

def sobel(img, new_img_arr):
    filter_shape = 3

    gradient_hor = np.asarray([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]])

    gradient_ver = np.asarray([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            temp = new_img_arr[i:i + filter_shape, j:j + filter_shape]
            res_x = np.sum(temp * gradient_hor)
            res_y = np.sum(temp * gradient_ver)
            img[i, j] = np.sqrt(np.square(res_x) + np.square(res_y))

    return img.astype(np.uint8)


if __name__ == '__main__':
    flower = cv2.imread('../Sample Images/img_11.png')
    flower = cv2.resize(flower, (800, 600))

    flower_gray = cv2.cvtColor(flower, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Before filter", flower_gray)

    new_img_arr = padding(flower_gray)
    new_img_arr = sobel(flower_gray, new_img_arr)
    cv2.imshow("After filter", new_img_arr)
    print(new_img_arr)
    cv2.waitKey(0)
