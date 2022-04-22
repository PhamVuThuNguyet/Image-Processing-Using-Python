import numpy as np
import cv2.cv2 as cv2


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y


def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))


# get 8 near point to decide where to grow
def selectConnects():
    connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1), Point(-1, 1),
                Point(-1, 0)]
    return connects


def regionGrow(img, seeds, thresh):
    m, n = img.shape
    seedMark = np.zeros([m, n])
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects()
    while (len(seedList) > 0):
        currentPoint = seedList.pop(0)
        seedMark[currentPoint.x, currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= m or tmpY >= n:
                continue

            grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
            if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = label
                seedList.append(Point(tmpX, tmpY))
    return seedMark


if __name__ == "__main__":
    img = cv2.imread('../Sample Images/img_4.png')
    img = cv2.resize(img, (800, 600))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Before filter", img)
    threshold = 5
    seeds = [Point(10, 10), Point(300, 400), Point(100, 300)]
    new_img_arr = regionGrow(img, seeds, threshold)
    cv2.imshow("After filter", new_img_arr)
    cv2.waitKey(0)
