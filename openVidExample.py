import numpy as np
import cv2 as cv

cap = cv.VideoCapture('images/mANDmvideo.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    

    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
