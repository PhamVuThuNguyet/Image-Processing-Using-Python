import cv2.cv2 as cv2
import numpy as np


def logarithmic(img, c):
    return (float(c) * np.log(1.0 + img)).astype(np.uint8)


if __name__ == "__main__":
    flower = cv2.imread('../Sample Images/img_3.png')
    flower = cv2.resize(flower, (800, 600))

    img_shape = flower.shape

    flower_gray = cv2.cvtColor(flower, cv2.COLOR_BGR2GRAY)
    print(flower_gray)
    cv2.imshow("Before filter", flower_gray)

    c = 255.0
    flower_gray_new = logarithmic(flower_gray, c)
    print(flower_gray_new)
    cv2.imshow("After filter", flower_gray_new)
    cv2.waitKey(0)
