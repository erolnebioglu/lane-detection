import cv2
import numpy as np

path = "testVideos/road.mp4"

cap = cv2.VideoCapture(path)
clicked_points = []
color = (0,255,0)
font = cv2.FONT_HERSHEY_SIMPLEX


ret,img = cap.read()

img = cv2.resize(img,(640,480))

def click_event(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        ##print("coordinates({},{}):".format(x,y))
        cv2.circle(img,(x,y),5,color,-1)
        cv2.putText(img,f'({x},{y})',(x,y-10),font,0.5,color,2)
        cv2.imshow("test",img)
        clicked_points.append((x,y))



cv2.imshow("test",img)

cv2.setMouseCallback("test",click_event)



if cv2.waitKey(0) == 27:
    cv2.imwrite("coordinate.png",img)
    for point in clicked_points:
        print(f"coordinates:{point}")

cv2.destroyAllWindows()