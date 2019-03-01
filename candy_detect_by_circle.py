import cv2 as cv
import numpy as np

img = cv.imread('candyBigSmallerTiny.jpg',1)
output=img.copy()
def get_bgr(event,x,y,flags, params):
        global mouseX,mouseY
        if event==cv.EVENT_LBUTTONDOWN:
            mouseX,mouseY=x,y
            print("BGR value")
            print(img1[y,x])
def simplify_img(img, channels, depth=3):
    blur = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT)
    Z = blur.reshape((-1, depth))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(Z, channels, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(img.shape)
    return res2
blur1 = cv.GaussianBlur(img, (3,3),cv.BORDER_DEFAULT)
hsv = cv.cvtColor(blur1, cv.COLOR_BGR2HSV)
h, s, v = cv.split(hsv)
# simplify each channel to 2 values, this becomes 8 combinations once channels are recombined
k = 5
h = simplify_img(h, k, 1)
s = simplify_img(s, k, 1)
v = simplify_img(v, k, 1)
hsv_ch = [h, s, v]

# convert back to BGR, colors become distorted
img1 = cv.cvtColor(cv.merge(hsv_ch), cv.COLOR_HSV2BGR)

#setting boundaries for detecting candies.  Some need more than one set of boundaries
green_lower=np.array([57, 142, 15],np.uint8)
green_upper=np.array([ 97, 237, 27],np.uint8)

orange_lower=np.array([ 25, 102, 236],np.uint8)
orange_upper=np.array([ 137, 173, 236],np.uint8)

blue_lower=np.array([236, 152, 26],np.uint8)
blue_upper=np.array([236, 152, 26],np.uint8)

red1_lower=np.array([116, 56, 143],np.uint8)
red1_upper=np.array([210, 106, 236],np.uint8)

red2_lower=np.array([206, 137, 236],np.uint8)
red2_upper=np.array([216, 171, 236],np.uint8)

brown1_lower=np.array([58, 48, 56],np.uint8)
brown1_upper=np.array([106, 100, 90],np.uint8)

brown2_lower=np.array([48, 67, 39],np.uint8)
brown2_upper=np.array([48, 67, 39],np.uint8)

brown3_lower=np.array([143, 127, 103],np.uint8)
brown3_upper=np.array([143, 127, 103],np.uint8)

brown4_lower=np.array([102, 90, 107],np.uint8)
brown4_upper=np.array([102, 90, 107],np.uint8)

yellow_lower=np.array([20, 183, 172],np.uint8)
yellow_upper=np.array([92, 236, 226],np.uint8)

#making a mask the boundaries for all the colors
green = cv.inRange(img1,green_lower,green_upper)
orange = cv.inRange(img1,orange_lower,orange_upper)
blue = cv.inRange(img1,blue_lower,blue_upper)
red1 = cv.inRange(img1,red1_lower,red1_upper)
red2 = cv.inRange(img1,red2_lower,red2_upper)
brown1 = cv.inRange(img1,brown1_lower,brown1_upper)
brown2 = cv.inRange(img1,brown2_lower,brown2_upper)
brown3 = cv.inRange(img1,brown3_lower,brown3_upper)
brown4 = cv.inRange(img1,brown4_lower,brown4_upper)
yellow = cv.inRange(img1,yellow_lower,yellow_upper)

#recombining any that needed more than one boundary
red = cv.add(red1,red2)
brownish = cv.add(brown1,brown2)
brownish1 = cv.add(brownish,brown3)
brown = cv.add(brownish1,brown4)

kernal2 = np.ones((3,3),np.uint8)

#enlarging mask
green = cv.dilate(green,kernal2)
orange = cv.dilate(orange,kernal2)
blue = cv.dilate(blue,kernal2)
red = cv.dilate(red,kernal2)
brown = cv.dilate(brown,kernal2)
yellow = cv.dilate(yellow,kernal2)

#combining all colors onto one image
m=cv.add(green,orange)
m1=cv.add(m,blue)
m2=cv.add(m1,red)
m3=cv.add(m2,brown)
mask=cv.add(m3,yellow)

cv.imshow('mask',mask)
#this and the next block are just to see the colors from the mask
test=cv.bitwise_and(img1,img1,mask=green)
test1=cv.bitwise_and(img1,img1,mask=orange)
test2=cv.bitwise_and(img1,img1,mask=blue)
test3=cv.bitwise_and(img1,img1,mask=red)
test4=cv.bitwise_and(img1,img1,mask=brown)
test5=cv.bitwise_and(img1,img1,mask=yellow)

add=cv.add(test,test1)
add1=cv.add(add,test2)
add2=cv.add(add1,test3)
add3=cv.add(add2,test4)
final=cv.add(add3,test5)

cv.imshow('final',final)
params=cv.SimpleBlobDetector_Params()

params.filterByArea=True
params.maxArea=400

ver = (cv.__version__).split('.')
if int(ver[0])<3:
    detector = cv.SimpleBlobDetector(params)
else:
    detector = cv.SimpleBlobDetector_create(params)

keypoints = detector.detect(final)

##final_with_keypoints =cv.drawKeypoints(im, keypoints, np.array([]), (255,0,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
objects=[]
##for k in keypoints:
##    objects.append(Rect(int(k.pt[0]-k.size),int(k.pt[1]-k.size),int(k.size*2),int(k.size*2)))
##cv.imshow("keypoints", final_with_keypoints)

##(contours,hierarchy)=cv.findContours(red,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
##	
##for contour in contours:
##    area = cv.contourArea(contour)
##    if area<300:
##        x,y,w,h = cv.boundingRect(contour)
##        img = cv.rectangle(img,(x,y),(x+100,y+100),(0,0,255),2)
##        cv.putText(output,"RED color",(x,y),cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255))


gray = cv.cvtColor(final,cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray,(3,3),cv.BORDER_DEFAULT)
g=np.median(blur)
lower=int(max(0,(1.0-.33)*g))
upper=int(min(255,(1.0-.33)*g))
canny = cv.Canny(mask, lower,upper)
kernel = np.ones((3,3),np.uint8)
kernel1 = np.ones((1,1),np.uint8)
dilate = cv.dilate(canny,kernel,iterations=1)
erode = cv.erode(dilate, kernel1, iterations=3)

circles = cv.HoughCircles(erode,cv.HOUGH_GRADIENT,1.2,10,
                          param1=50, param2=30, minRadius=0, maxRadius=30)
if circles is not None:
    circles = np.round(circles[0,:]).astype("int")
    for (x,y,r)in circles:
##        blue=img[y,x,0]
##        green=img[y,x,1]
##        red=img[y,x,2]
##        print(blue)
        cv.circle(output,(x,y),r,(0,255,0),2)
            

cv.imshow('img1', img1)
cv.imshow('edge', erode)
cv.imshow('image',img)
cv.imshow('output',output)
cv.setMouseCallback('img1',get_bgr,param=img1)
k = cv.waitKey(0)
cv.destroyAllWindows()
