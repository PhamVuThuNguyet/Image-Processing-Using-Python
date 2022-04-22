import numpy as np
from tqdm import tqdm

import cv2.cv2 as cv2

np.random.seed(42)


def init_centroid(img, k):
    r, c = img.shape
    centroid_arr = np.empty([k, c])
    for i in range(k):
        rand_num = np.random.randint(r)
        centroid_arr[i] = img[rand_num]

    return centroid_arr.astype(np.uint8)


def distance(x1, x2):
    x1_square_sum = np.sum(np.square(x1), axis=1)
    x2_square_sum = np.sum(np.square(x2), axis=1)

    mul = np.dot(x1, x2.T)
    dist = np.sqrt(abs(x1_square_sum[:, np.newaxis] + x2_square_sum - 2 * mul))

    return dist


def choosing_cluster(img, centroid):
    r, c = img.shape
    cluster_idx = np.empty([r])
    dist = distance(img, centroid)
    cluster_idx = np.argmin(dist, axis=1)

    return cluster_idx


def update_centroid(old_centroid, cluster_idx, img):
    k, c = old_centroid.shape
    new_centroid = np.empty(old_centroid.shape)
    for i in range(k):
        new_centroid[i] = np.mean(img[cluster_idx == i], axis=0)

    return new_centroid


if __name__ == '__main__':
    img = cv2.imread('../Sample Images/img_4.png')
    img = cv2.resize(img, (800, 600))
    img_shape = img.shape
    cv2.imshow("Before filter", img)
    img = img.reshape(img_shape[0] * img_shape[1], img_shape[2])

    k = 2
    centroid = init_centroid(img, k)

    for i in range(100):
        cluster_idx = choosing_cluster(img, centroid)
        centroid = update_centroid(centroid, cluster_idx, img)

    update_img_arr = np.copy(img)
    for i in range(k):
        indices_current_cluster = np.where(cluster_idx == i)[0]
        update_img_arr[indices_current_cluster] = centroid[i]

    update_img_arr = update_img_arr.reshape(img_shape)
    cv2.imshow("After filter", update_img_arr)
    cv2.waitKey(0)
