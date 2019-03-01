import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
cv.namedWindow("Video")
cv.namedWindow("HSV")
cv.namedWindow("Tracking")
cv.namedWindow("dials")

cap.set(cv.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 300)
# create scalars
min_scalars = np.zeros(3)
max_scalars = np.zeros(3)
max_scalars[0] = 180
max_scalars[1] = 255
max_scalars[2] = 255


def change_slider_maxH(value):
    max_scalars[0] = value


def change_slider_maxS(value):
    max_scalars[1] = value


def change_slider_maxV(value):
    max_scalars[2] = value


def change_slider_minH(value):
    min_scalars[0] = value


def change_slider_minS(value):
    min_scalars[1] = value


def change_slider_minV(value):
    min_scalars[2] = value


# prints out the x and y location of a mouse click
def mouseCall(evt, x, y, flags, pic):
    if evt == cv.EVENT_LBUTTONDOWN:
        print("[" + str(pic[y][x][0]) + ", " + str(pic[y][x][1]) + ", " + str(pic[y][x][2]) + "],")


cv.createTrackbar("max B", "dials", 255, 255, change_slider_maxH)
cv.createTrackbar("max G", "dials", 255, 255, change_slider_maxS)
cv.createTrackbar("max R", "dials", 255, 255, change_slider_maxV)
cv.createTrackbar("min B", "dials", 0, 255, change_slider_minH)
cv.createTrackbar("min G", "dials", 0, 255, change_slider_minS)
cv.createTrackbar("min R", "dials", 0, 255, change_slider_minV)

while True:
    # get video info
    img = cv.imread("imagesWOvideo/candyBigSmallerTiny.jpg", cv.IMREAD_COLOR)
    img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)

    # set HSV image BGR2HLS (NB 87 S>|; B 23 H >|), BGR2LUV (B V><)
    #hsv = cv.cvtColor(img, cv.COLOR_BGR2LUV)
    hsv = img.__copy__()
    cv.setMouseCallback("HSV", mouseCall, hsv)

    # create track bars
    # min HSV = 0, 72, 150 - brown is lost
    tracking_img = cv.inRange(hsv, min_scalars, max_scalars)
    # remove image noise
    tracking_img = cv.erode(tracking_img, np.ones((6, 6)))
    tracking_img = cv.dilate(tracking_img, np.ones((6, 6)))

    # show results
    #cv.imshow("Video", img)
    cv.imshow("HSV", hsv)
    #cv.imshow("Tracking", tracking_img)

    k = cv.waitKey(1)
    # this is the "esc" key
    if k == 27:
        break
cv.destroyAllWindows()