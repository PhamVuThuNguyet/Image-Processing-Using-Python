import cv2.cv2 as cv2
import numpy as np

def thresholding(img, threshold):
    return ((img > threshold) * 255).astype(np.uint8)

if __name__ == "__main__":
    flower = cv2.imread('../Sample Images/img_1.png')
    flower = cv2.resize(flower, (800, 600))

    img_shape = flower.shape

    flower_gray = cv2.cvtColor(flower, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Before filter", flower_gray)

    threshold = np.mean(flower_gray)
    flower_gray_new = thresholding(flower_gray, threshold)
    print(flower_gray_new)
    cv2.imshow("After filter", flower_gray_new)
    cv2.waitKey(0)