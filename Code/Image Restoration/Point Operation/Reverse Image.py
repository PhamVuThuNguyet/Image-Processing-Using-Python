import cv2.cv2 as cv2
import numpy as np

def reverse_image(img, intensity_max):
    return intensity_max - img

if __name__ == "__main__":
    flower = cv2.imread('../Sample Images/Flower.jpg')
    flower = cv2.resize(flower, (800, 600))

    img_shape = flower.shape

    flower_gray = cv2.cvtColor(flower, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Before filter", flower_gray)

    intensity_max = np.amax(flower_gray)
    flower_gray_new = reverse_image(flower_gray, intensity_max)

    cv2.imshow("After filter", flower_gray_new)
    cv2.waitKey(0)




