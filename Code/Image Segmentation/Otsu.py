import cv2.cv2 as cv2
import numpy as np


def otsu(img):
    var = 0
    M, N = img.shape
    mG = np.mean(img)

    for thres in range(256):
        sum_intensity_A = 1
        sum_intensity_B = 1
        sum_pixel_A = 1
        sum_pixel_B = 1
        for i in range(M):
            for j in range(N):
                if (img[i, j] >= thres):  # A group
                    sum_pixel_A = sum_pixel_A + 1
                    sum_intensity_A = sum_intensity_A + img[i, j]
                else:
                    sum_pixel_B = sum_pixel_B + 1
                    sum_intensity_B = sum_intensity_B + img[i, j]

        P1 = sum_pixel_A / (M * N)
        P2 = sum_pixel_B / (M * N)
        m1 = sum_intensity_A / sum_pixel_A
        m2 = sum_intensity_B / sum_pixel_B
        var_temp = P1 * ((m1 - mG) ** 2) + P2 * ((m2 - mG) ** 2)

        if (var_temp > var): # find max var to find optimus threshold
            var = var_temp
            opt_thres = thres

    return opt_thres


def segmentation(img, threshold):
    m, n = img.shape
    new_img_arr = np.zeros([m, n])
    for i in range(m):
        for j in range(n):
            if (img[i, j] < threshold):
                new_img_arr[i, j] = 0
            else:
                new_img_arr[i, j] = 225
    return new_img_arr


if __name__ == "__main__":
    img = cv2.imread('../Sample Images/img_4.png')
    img = cv2.resize(img, (800, 600))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Before filter", img)
    threshold = otsu(img)
    new_img_arr = segmentation(img, threshold)
    cv2.imshow("After filter", new_img_arr)
    cv2.waitKey(0)
