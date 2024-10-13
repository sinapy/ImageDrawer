import turtle

import cv2 as cv
import numpy as np
import time

import rgb as rgb
from scipy import ndimage

before_finish = time.time()

img = cv.imread('school.jpg')
if img is None:
    img = cv.imread('picture.png')
if img is None:
    img = cv.imread('picture.jpeg')

sc = turtle.Screen()

sc.setup(1400, 1299)

# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# lower = np.array([0, 71, 0], dtype="uint8")
# upper = np.array([179, 255, 255], dtype="uint8")
# mask = cv.inRange(hsv, lower, upper)


shape = img.shape
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_gaussian = cv.GaussianBlur(gray, (3, 3), 0);
kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
# img_prewittx = cv.filter2D(img_gaussian, -1, kernelx)
# img_prewitty = cv.filter2D(img_gaussian, -1, kernely)
canny = cv.Canny(img, 500, 250)
# canny = img_prewittx

contours, _ = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
ym = max(c[:, :, 1].max() for c in contours)
for c in contours:
    c[:, :, 1] = ym - c[:, :, 1]
sc.tracer(0)
pixel = turtle.Turtle(visible=False)


# Find closest contour.  We'll use Manhattan distance
# because it won't really matter.

def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def findClosest(contours, ept):
    dist = []
    for c in contours:
        dist.append(distance(*c[0], ept))
        dist.append(distance(*c[-1], ept))

    return dist.index(min(dist)) // 2


# Draw in a forward direction.

def drawContour(ctr):
    n = 1
    if len(ctr) == 1:
        return ctr[0][0]
    pixel.penup()
    pixel.goto(ctr[0][0][0] / n - shape[0] / (n * 2) - 100, ctr[0][0][1] / n - shape[1] / (n * 2) + 50)
    # pixel.goto(*ctr[0])
    pixel.pendown()
    for pt in ctr[1:]:
        # pixel.goto(*pt[0])
        pixel.goto(pt[0][0] / n - shape[0] / (n * 2) - 100, pt[0][1] / n - shape[1] / (n * 2) + 50)

    return pt[0] / n


nxt = drawContour(contours[0])
contours = contours[1:]
while contours:
    ndx = findClosest(contours, nxt)
    nxt = drawContour(contours[ndx])
    contours = contours[0:ndx] + contours[ndx + 1:]
    sc.update()

after_finish = time.time()
print(f'{after_finish - before_finish} seconds')
# Update the screen to see the changes


# Keep the window open
sc.mainloop()
