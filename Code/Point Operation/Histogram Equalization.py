import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt


def hist_equalize(img):
    hist, _ = np.histogram(img, 256, [0, 255])

    # normalized histogram
    hist = hist / (img.shape[0] * img_shape[1])

    # cumulative distribution function
    cdf = np.cumsum(hist)

    # mask all 0 values in cumulative sum array
    cdf_m = np.ma.masked_equal(cdf, 0)

    # calculate K: K(i)  = (Z(i) - min(Z)) * 255 / (max(Z) - min(Z))
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())

    # masked places are 0
    return np.ma.filled(cdf_m, 0).astype(np.uint8)


if __name__ == "__main__":
    flower = cv2.imread('../Sample Images/img_5.png')
    flower = cv2.resize(flower, (800, 600))

    img_shape = flower.shape

    flower_gray = cv2.cvtColor(flower, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Before filter", flower_gray)

    flower_gray_new = hist_equalize(flower_gray)[flower_gray.flatten()]
    flower_gray_new = np.reshape(flower_gray_new, flower_gray.shape)
    print(flower_gray_new)

    cv2.imshow("After filter", flower_gray_new)

    fig1 = plt.figure()
    fig2 = plt.figure()

    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)

    ax1.hist(flower_gray_new.ravel(), 256, [0, 256]);
    ax2.hist(flower_gray.ravel(), 256, [0, 256]);

    ax1.set_title("After")
    ax2.set_title("Before")

    plt.show();
    cv2.waitKey(0)
