import cv2
import imutils
import numpy as np


color = (0,255,0)




cap = cv2.VideoCapture("testVideos/road.mp4")


ret,frame = cap.read()


top_left = (236,403)
bottom_left = (173,465)
top_right = (381,403)
bottom_right = (431,465) 

pts1 = np.float32([top_left,bottom_left,top_right,bottom_right])
pts2 = np.float32([[0,0],[0,480],[640,0],[640,480]])

if ret == True:
    print("[INFO].. Shape",frame.shape)
    
else:
    print("[INFO]... the video is not loaded succesfully")
 
def nothing (x):
    pass

cv2.namedWindow("trackbars")

cv2.createTrackbar("l-h","trackbars",0,255,nothing)
cv2.createTrackbar("l-s","trackbars",0,255,nothing)
cv2.createTrackbar("l-v","trackbars",200,255,nothing)

cv2.createTrackbar("u-h","trackbars",255,255,nothing)
cv2.createTrackbar("u-s","trackbars",50,255,nothing)
cv2.createTrackbar("u-v","trackbars",255,255,nothing)

 
 
while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(640,480))
    
    frame_copy = frame.copy()
    cv2.circle(frame_copy,top_left,5,color,-1)
    cv2.circle(frame_copy,bottom_left,5,color,-1)
    cv2.circle(frame_copy,top_right,5,color,-1)
    cv2.circle(frame_copy,bottom_right,5,color,-1)
    
    
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    
    transformedFrame = cv2.warpPerspective(frame,matrix,(640,480))    
    
    transformedFrame_hsv = cv2.cvtColor(transformedFrame,cv2.COLOR_BGR2HSV)    
                 
    
    LOWER_H = cv2.getTrackbarPos("l-h","trackbars")
    LOWER_S = cv2.getTrackbarPos("l-s","trackbars")
    LOWER_V = cv2.getTrackbarPos("l-v","trackbars")
    
    UPPER_H = cv2.getTrackbarPos("u-h","trackbars")
    UPPER_S = cv2.getTrackbarPos("u-s","trackbars")
    UPPER_V = cv2.getTrackbarPos("u-v","trackbars")
    
    LOWER = np.array([LOWER_H,LOWER_S,LOWER_V])
    UPPER = np.array([UPPER_H,UPPER_S,UPPER_V])
    
    mask = cv2.inRange(transformedFrame_hsv,LOWER,UPPER)
    
    
    
    histogram = np.sum(mask[mask.shape[0]//2:, :],axis=0)
    midpoint = np.int32(histogram.shape[0]/2)
    leftSide = np.argmax(histogram[:midpoint])
    rightSide = np.argmax(histogram[midpoint:])+ midpoint
    
    left_x = []
    right_x = []
    
    mask_copy = mask.copy()
    starting_y = 480
    while starting_y>0:
        img = mask[starting_y-40:starting_y,leftSide-50:leftSide+50]
        
        contours,_ = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"]!=0:
                center_x  = np.int32(M["m10"]/M["m00"])
                center_y  = np.int32(M["m01"]/M["m00"])
                leftSide = leftSide - 50 + center_x
                
                
        img = mask[starting_y-40:starting_y,rightSide-50:rightSide+50]
        
        contours,_ = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"]!=0:
                center_x  = np.int32(M["m10"]/M["m00"])
                center_y  = np.int32(M["m01"]/M["m00"])
                rightSide = rightSide - 50 + center_x       
                
        
        
        cv2.rectangle(mask_copy,(leftSide-60,starting_y),(leftSide+60,starting_y-40),(255,255,255),2)
        cv2.rectangle(mask_copy,(rightSide-60,starting_y),(rightSide+60,starting_y-40),(255,255,255),2)
        
        
        starting_y = starting_y - 40
    #cv2.imshow("frame",frame_copy)
    #cv2.imshow("birds eye view-bgr",transformedFrame)
    #cv2.imshow("birds eye view-hsv",transformedFrame_hsv)
    #cv2.imshow("birds eye view-mask",mask)
    cv2.imshow("slidingWindow",mask_copy)
    if cv2.waitKey(0)==27:
        break
   
cv2.destroyAllWindows()        
