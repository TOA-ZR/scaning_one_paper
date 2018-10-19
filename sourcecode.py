import cv2
import numpy as np
import math



#load an image
path = r"C:\Users\jo\Desktop\Pic\IMG_9374.JPG"
#path = r"20131030153346984.jpg"
imgcol=cv2.imread(path)
#meanshift方法 有点慢
#imgcol = cv2.pyrMeanShiftFiltering(imgcol, 25, 10)
img_colorP = np.copy(imgcol)
height,width = imgcol.shape[:2]

#imgray = cv2.imread(path,0)
drawing = np.zeros(imgcol.shape,np.uint8)     # Image to draw the contours
drawing2 = np.zeros(imgcol.shape,np.uint8)
imgray = cv2.cvtColor(imgcol,cv2.COLOR_BGR2GRAY)
#cv2.imshow("imgcol",imgcol)
#cv2.imshow("img_gray",imgray)
#check if img_gray is loaded fine
if imgray is None:
        print("Error opening image!")



"""
preprocessing
1. Reduce noise by smoothing the image (Gaussian blur)
2. Fill holes and reduce clutter by morphological operation
3. Adaptive Thresholding after morphological operation 
"""
""" RGB2HSV
imgHSV = cv2.cvtColor(imgcol, cv2.COLOR_BGR2HSV)
imgbinar = cv2.inRange(imgHSV, (0, 0, 200), (180, 50, 255))
cv2.imshow("imgbinar",imgbinar)
"""

#1. Gaussian Blus

imgblurred= cv2.GaussianBlur(imgray,(3,3),0) #TODO
#imgblurred=cv2.copyMakeBorder(imgblurred,5,5,5,5,cv2.BORDER_CONSTANT,value=(255,255,255)) #TODO 可以去掉 画框框的
#imgblur= cv2.medianBlur(imgray,5)

#TODO 用原来的adaptive试一试
imgthresh = cv2.adaptiveThreshold(imgblurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,51,2)
#ret, imgthresh = cv2.threshold(imgblurred, 0, 200, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
#ret,thresh=cv2.threshold(imgblurred,127,255,0)
#cv2.imshow('thresh', imgthresh)
##TODO 图像增强，把边连起来，看mv的ppt
#imgconn=cv2.connectedComponents(imgthresh)


#2. Morphological Operation
#kernel = np.ones((5,5),np.uint8) #TODO
#imgmorpho = cv2.morphologyEx( imgthresh, cv2.MORPH_CLOSE, kernel) # input: binary image
#cv2.imshow("imgmorpho",imgmorpho)
#3. Adaptive Thresholding after Filtering
#img_binary = cv2.adaptiveThreshold(img_morphology,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,5)#TODO



""" paper detection algorithm
1. find border with Canny Detector
2. find contours in the segmented binary image
3. approxPolyDP """

#1. Canny Detector
thresh = 10
edges=cv2.Canny(imgthresh, thresh, thresh*10) #input gray image
#cv2.imshow("canny",edges)
#2. find_Contours
#Paramters of cv2.findContours Function in  opencv2 and opencv3 are different
#img_contour,contours, hierarchy= cv2.findContours( img_canny.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
img_contour,contours, hierarchy= cv2.findContours( edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#cv2.imshow('img_contour', img_contour)
cnts = contours[0]
#approxPoly contour
epsilon = 0.1*cv2.arcLength(cnts,True)
approx = cv2.approxPolyDP(cnts,epsilon,True)
img_contour=cv2.drawContours(img_contour, approx, -1, (0, 255, 0), 3)
#cv2.imshow("approx_color", img_contour)
#draw contour
img_gray=cv2.drawContours(imgray,approx,0,(0,255,0),3)
#cv2.imshow('approx', img_gray)
#get convexhull
area = cv2.contourArea(cnts)
perimeter = cv2.arcLength(cnts,True)
hull = cv2.convexHull(cnts)
#get corners
# 确保至少有一个轮廓被找到
if len(cnts) > 0:
#draw convexhull
        for cnt in contours:
                hull = cv2.convexHull(cnt)
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                approxsub = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
                corners = len(approxsub)
        # filter of extremely unlikely contours
                if (area >= 500 and perimeter > 800 and len(approxsub) == 4 ): #TODO
                        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 2)  # draw contours in green color
                        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 2)  # draw contours in red color

        #area
                        #cv2.namedWindow("input", 0);
                        #cv2.resizeWindow("input", 4032, 3024);
                        #cv2.namedWindow("output", 0);
                       # cv2.resizeWindow("output", 4032, 3024);
                        cv2.imshow('output', drawing)
                        cv2.imwrite("output.jpg", drawing)
                        cv2.imshow('input', drawing)

                        # TODO 把截图画出来 变成四边形

                        docCnt = approxsub
                        newimg = imgcol.copy()
                        # draw the corner on the original image
                        for i in docCnt:
                                # circle函数为在图像上作图，新建了一个图像用来演示四角选取
                                cv2.circle(newimg, (i[0][0], i[0][1]), 50, (0, 255, 0), -1)
                               # cv2.namedWindow("newimg", 0);
                               # cv2.resizeWindow("newimg", 4032, 3024);
                                cv2.imshow('newimg', newimg)
                                cv2.imwrite('newimg.jpg', newimg)

                        # map the image
                        """

                        pts2 = np.float32([[0, 0], [400, 0], [0, 200], [400, 200]])
                        for i in docCnt:
                                pts1.append([i[0][0], i[0][1]])

                                M = cv2.getPerspectiveTransform(pts1, pts2)
                                result = cv2.warpPerspective(drawing2, M, (0, 0))
                        """
                        #TODO 把坐标按照顺序放到pts1里，然后用M来perspective 完全的是语法问题了
                        m=0
                        a = []
                        for i in docCnt:
                                print(i)
                                a.append(i)
                             #   a[m]=i
                              #  m=m+1
                        #print(a)
                        pts1 = np.float32(a)
                        print(a)
                        #pts2 = np.float32([[0, 0], [200, 0], [0, 400], [200, 400]])
                        pts2 = np.float32([[width, 0], [0, 0], [0, height], [width, height]])

                        M = cv2.getPerspectiveTransform(pts1, pts2)
                        #result = cv2.warpPerspective(imgcol, M,(200,400))
                        result = cv2.warpPerspective(imgcol, M, (width, height))
                        cv2.imshow("result",result)
                        cv2.imwrite("result.jpg",result)








# Wait and Exit
cv2.waitKey(0)
cv2.destroyAllWindows()















