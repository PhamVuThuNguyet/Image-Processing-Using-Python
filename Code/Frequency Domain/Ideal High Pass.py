import cmath

import cv2.cv2 as cv2
import numpy as np
import math


def DFT_1D(img):
    M = len(img)
    new_img_arr = np.zeros(M, dtype=complex)
    for i in range(M):
        for j in range(M):
            new_img_arr[i] += (img[j] * np.exp(-1j * 2 * np.pi * i * j / M))
    return new_img_arr


def inverseDFT_1D(img):
    M = len(img)
    new_img_arr = np.zeros(M, dtype=complex)
    for i in range(M):
        for j in range(M):
            new_img_arr[i] += (img[j] * np.exp(1j * 2 * np.pi * i * j / M))
        new_img_arr[i] /= M
    return new_img_arr


def inverseDFT_2D(img):
    P, Q = img.shape

    for i in range(P):
        img[i] = inverseDFT_1D(img[i])
    for j in range(Q):
        img[:, j] = inverseDFT_1D(img[:, j])

    return img


def DFT_2D(img):
    P, Q = img.shape

    for i in range(P):
        img[i] = DFT_1D(img[i])
    for j in range(Q):
        img[:, j] = DFT_1D(img[:, j])

    return img


def ideal_highpass(D0, P, Q):
    H = np.zeros((P, Q))
    P0 = int(P / 2)
    Q0 = int(Q / 2)

    for u in range(P):
        for v in range(Q):
            if (np.sqrt(np.square(u - P0) + np.square(v - Q0))) <= D0:
                H[u, v] = 0
            else:
                H[u, v] = 1
    return H


def frequency_filter(img):
    M, N = img.shape

    # padding size
    P = 2 * M
    Q = 2 * N

    # create padded image
    new_img_arr = np.zeros((P, Q))
    new_img_arr[:M, :N] = img

    # center the Fourier transform
    for i in range(P):
        for j in range(Q):
            new_img_arr[i, j] = new_img_arr[i, j] * np.power(-1, i + j)

    # DFT
    F = DFT_2D(new_img_arr)

    # Ideal high pass filter
    D0 = 10
    H = ideal_highpass(D0, P, Q)
    print(H)

    # Calculate the product
    G = np.multiply(H, F)

    # reverse transform
    g = inverseDFT_2D(G)

    g = np.asarray(g.real)
    for i in range(P):
        for j in range(Q):
            g[i, j] = g[i, j] * np.power(-1, i + j)

    # extract mxn region
    g = g[:M, :N]

    return g


if __name__ == '__main__':
    flower = cv2.imread('../Sample Images/img_6.png')
    flower = cv2.resize(flower, (800, 600))

    img_shape = flower.shape

    flower_gray = cv2.cvtColor(flower, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Before filter", flower_gray)

    new_img_arr = frequency_filter(flower_gray)
    new_img_arr.astype(np.uint8)
    cv2.imshow("After filter", new_img_arr)
    cv2.waitKey(0)
