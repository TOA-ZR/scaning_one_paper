import cv2 as cv

import numpy as np
path = r"C:\Users\jo\Desktop\IMG_9375.JPG"
img = cv.imread(path)

# preprocessing

img = cv.GaussianBlur(img,(5,5),0)
cv.imwrite('save11.jpg',img)


#image processing
img_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
img_threshold = cv.inRange(img_HSV, (0, 0, 221), (180, 30, 255))
cv.imwrite('save12.jpg',img_threshold)

                                           










