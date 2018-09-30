import cv2
import numpy as np
import math

#load an image
path = r"C:\Users\jo\Desktop\Pic\IMG_9374.JPG"
img_color=cv2.imread(path)
img_colorP = np.copy(img_color)
img_gray = cv2.imread(path,0)
#img_gray = cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)
cv2.imshow("img_gray",img_gray)
#check if img_gray is loaded fine
if img_gray is None:
        print("Error opening image!")

"""
preprocessing
1. Reduce noise by smoothing the image (Gaussian blur)
2. Fill holes and reduce clutter by morphological operation
3. Adaptive Thresholding after 
"""
#1. Gaussian Blus
img_gauss= cv2.GaussianBlur(img_gray,(5,5),0) #TODO
#2. Morphological Operation
kernel = np.ones((5,5),np.uint8) #TODO
img_morphology = cv2.morphologyEx( img_gauss, cv2.MORPH_CLOSE, kernel) # input: binary image
#3. Adaptive Thresholding after Filtering
img_binary = cv2.adaptiveThreshold(img_morphology,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,5)#TODO


"""
paper detection algorithm
1. find border with Canny Detector
2. find contours in the segmented binary image
3. HoughLines to find contours
"""
#1 Canny Detector
img_canny=cv2.Canny(img_binary, 200, 255) #input gray image
cv2.imshow("canny",img_canny)
#2. find_Contours
img_contour, contours, hierarchy = cv2.findContours(img_canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_gray,contours,-1,(0,255,0),3) #直接在原图上修改
cv2.imshow('img_contour.jpg',img_gray)
cv2.imshow('img_contour', img_contour)
cv2.imwrite("result.jpg",img_gray)


# Wait and Exit
cv2.waitKey(0)
cv2.destroyAllWindows()















