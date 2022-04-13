import cv2.cv2 as cv2
import numpy as np


def gray_level_slicing(img, min_range, max_range):
    return (((img > min_range) & (img < max_range)) * 255).astype(np.uint8)


if __name__ == "__main__":
    flower = cv2.imread('../Sample Images/img_7.png')
    flower = cv2.resize(flower, (800, 600))

    img_shape = flower.shape

    flower_gray = cv2.cvtColor(flower, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Before filter", flower_gray)

    min_range = 30
    max_range = 100
    flower_gray_new = gray_level_slicing(flower_gray, min_range, max_range)
    print(flower_gray_new)
    cv2.imshow("After filter", flower_gray_new)
    cv2.waitKey(0)
