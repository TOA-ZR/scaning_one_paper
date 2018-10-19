import cv2
import numpy as np

"""
Load the image
1.Load  image
2.Get Shape of image
3. Convert BGR to gray
4. create 2 empty image same size as input image, for next process(draw the contours)
"""
#1.Load  image
path = r"C:\Users\jo\Desktop\Pic\IMG_9374.JPG"
imgcol=cv2.imread(path)
img_colorP = np.copy(imgcol)

#2.Get Shape of image
height,width = imgcol.shape[:2]

#3. Convert BGR to gray
imgray = cv2.cvtColor(imgcol,cv2.COLOR_BGR2GRAY)

#4. create 2 empty image
drawing = np.zeros(imgcol.shape,np.uint8)
drawing2 = np.zeros(imgcol.shape,np.uint8)


if imgray is None:
        print("Error opening image!")
else:
        cv2.imshow("imgray", imgray)




"""
Pre-processing
1. Gaussian blur (Reduce noise by smoothing the image)
2. Adaptive Thresholding for binary image
3. Morphological Operation(CLOSE) to Fill holes and reduce clutter 
"""

#1. Gaussian Blus
imgblurred= cv2.GaussianBlur(imgray,(3,3),0) #TODO

#2. Adaptive Thresholding for binary image
imgthresh = cv2.adaptiveThreshold(imgblurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,51,2)

#3. Morphological Operation(CLOSE) to Fill holes and reduce clutter
kernel = np.ones((5,5),np.uint8) #TODO
imgthresh = cv2.morphologyEx( imgthresh, cv2.MORPH_CLOSE, kernel) # input: binary image
cv2.imshow("imgthresh",imgthresh)


""" 
PaperDetection Algorithm
1. CannyDetector to find edges (Input must be GrayValue image)
2. FindContours in the edge Image (Input: binary images for better accuracy)
3. Contour Approximation to obtain quadrilateral (Optimization)
4. Obtain a variety of characters of Contour
5. Convexhull Generation
6. Filter of extremely unlikely contours Generation
7.Draw the corner on the original image
8.Mapping the detected Paper from quadrilateral to rectangle(cropWarp)
"""

#1. CannyDetector to find edges
thresh = 10
edges=cv2.Canny(imgthresh, thresh, thresh*10)

#2. FindContours in the edge Image
imgcontour,contours, hierarchy= cv2.findContours(edges.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = contours[0]

#3. Contour Approximation to obtain quadrilateralr (optimization)
epsilon = 0.1*cv2.arcLength(cnts,True)
approx = cv2.approxPolyDP(cnts,epsilon,True)
#draw Contour Approximation
cv2.drawContours(imgcontour, approx, -1, (0, 255, 0), 3)
cv2.drawContours(imgray,approx,-1,(0,255,0),3)

#4. Obtain a variety of characters of Contours
area = cv2.contourArea(cnts) #area
perimeter = cv2.arcLength(cnts,True) #Perimeter

#5. Convexhull Generation
hull = cv2.convexHull(cnts)


# Make sure at least one contour will be found
if len(cnts) > 0:
#draw convexhull
        for cnt in contours:
                hull = cv2.convexHull(cnt)
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                approxsub = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
                corners = len(approxsub)

                # 6. Filter of extremely unlikely contours Generation
                if ( area >= 500 and perimeter > 800 and len(approxsub) == 4 ): #TODO
                        imgcnt=cv2.drawContours(imgcol.copy(), [cnt], 0, (0, 255, 0), 2)  # draw contours in green color
                        imghull=cv2.drawContours(imgcol.copy(), [hull], 0, (255, 0, 0), 2)  # draw contours in blue color
                        #draw input and output image (from Contours to Convexhull )
                        cv2.imshow('imgcnt', imgcnt)
                        cv2.imwrite("imgcnt.jpg", imgcnt)
                        cv2.imshow('imghull', imghull)
                        cv2.imwrite("imghull.jpg", imghull)

                        # 7.Draw the corner on the original image
                        docCnt = approxsub
                        newimg = imgcol.copy()

                        for i in docCnt:
                                # circle函数为在图像上作图，新建了一个图像用来演示四角选取
                                cv2.circle(newimg, (i[0][0], i[0][1]), 50, (0, 255, 0), -1)
                                cv2.imshow('cornerimg', newimg)
                                cv2.imwrite('cornerimg.jpg', newimg)

                        # 8.Mapping the detected Paper from quadrilateral to rectangle(cropWarp)
                        pts1 = []
                        for i in docCnt:
                                pts1.append(i)

                        pts1 = np.float32(pts1)
                        pts2 = np.float32([[width, 0], [0, 0], [0, height], [width, height]])

                        M = cv2.getPerspectiveTransform(pts1, pts2)
                        result = cv2.warpPerspective(imgcol, M, (width, height))
                        cv2.imshow("result",result)
                        cv2.imwrite("result.jpg",result)



# Wait and Exit
cv2.waitKey(0)
cv2.destroyAllWindows()