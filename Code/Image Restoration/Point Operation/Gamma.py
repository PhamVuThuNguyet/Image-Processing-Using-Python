import cv2.cv2 as cv2
import numpy as np

def normalize(img):
    m = np.amax(img)
    return img / m

def gamma_transform(img, c, gamma):
    return (float(c) * np.exp(np.log(img) * float(gamma))).astype(np.uint8)

if __name__ == "__main__":
    flower = cv2.imread('../Sample Images/img_4.png')
    flower = cv2.resize(flower, (800, 600))

    img_shape = flower.shape

    flower_gray = cv2.cvtColor(flower, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Before filter", flower_gray)

    c = 255
    gamma = 0.5
    flower_gray_new = gamma_transform(normalize(flower_gray), c, gamma)
    print(flower_gray_new)
    cv2.imshow("After filter", flower_gray_new)
    cv2.waitKey(0)




