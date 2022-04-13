import cmath

import cv2.cv2 as cv2
import numpy as np
import math


def DFT(img):
    M, N = img.shape
    L = len(img)

    temp = np.zeros((M, N), complex)
    new_img_arr = np.zeros((M, N), complex)

    m = np.arange(M)
    n = np.arange(N)
    x = m.reshape((M, 1))
    y = n.reshape((N, 1))

    for row in range(M):
        M1 = np.exp(-2j * np.pi * (n * y) / L)
        temp[row] = np.dot(M1, img[row])
    for col in range(N):
        M2 = np.exp(-2j * np.pi * (m * x) / L)
        new_img_arr[:, col] = np.dot(M2, temp[:, col])

    return new_img_arr


def gaussian_lowpass(D0, P, Q):
    H = np.zeros((P, Q))
    P0 = int(P / 2)
    Q0 = int(Q / 2)

    for u in range(P):
        for v in range(Q):
            H[u, v] = np.exp( -(np.square(u - P0) + np.square(v - Q0)) / (2 * np.power(D0, 2)))
    return H


def IDFT(img):
    M, N = img.shape
    L = len(img)

    temp = np.zeros((M, N), complex)
    new_img_arr = np.zeros((M, N), complex)

    m = np.arange(M)
    n = np.arange(N)
    x = m.reshape((M, 1))
    y = n.reshape((N, 1))

    for row in range(M):
        M1 = np.exp(2j * np.pi * (n * y) / L)
        temp[row] = np.dot(M1, img[row])
    for col in range(N):
        M2 = np.exp(2j * np.pi * (m * x) / L)
        new_img_arr[:, col] = np.dot(M2, temp[:, col])

    new_img_arr = new_img_arr / L

    return new_img_arr


def frequency_filter(img):
    M, N = img.shape

    # padding size
    P = 2 * M
    Q = 2 * N

    # create padded image
    new_img_arr = np.zeros((P, Q))
    new_img_arr[:M, :N] = img

    # Gaussian low pass filter
    D0 = 10
    H = gaussian_lowpass(D0, P, Q)
    print(H)

    # center the Fourier transform
    for i in range(P):
        for j in range(Q):
            new_img_arr[i, j] = new_img_arr[i, j] * np.power(-1, i + j)

    # DFT
    F = DFT(new_img_arr)

    # Calculate the product
    G = np.multiply(H, F)

    # reverse transform
    g = IDFT(G)

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
