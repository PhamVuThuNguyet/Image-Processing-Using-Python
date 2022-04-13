import cv2.cv2 as cv2
import numpy as np


def contrast_stretching(img):
    return (255.0 * (img - np.amin(img)) / (np.amax(img) - np.amin(img))).astype(np.uint8)


if __name__ == "__main__":
    flower = cv2.imread('../Sample Images/img_7.png')
    flower = cv2.resize(flower, (800, 600))

    img_shape = flower.shape

    flower_gray = cv2.cvtColor(flower, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Before filter", flower_gray)

    flower_gray_new = contrast_stretching(flower_gray)
    print(flower_gray_new)
    cv2.imshow("After filter", flower_gray_new)
    cv2.waitKey(0)
