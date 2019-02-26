import numpy as np
import cv2 as cv


def simplify_img(img, channels, depth=3):
    blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)
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


def get_unique_colors(img, is_non_color=False):
    # get all unique colors in the input image
    if is_non_color:
        x = 0
        y = 5000
        for row in img:
            tmp = max(row)
            if tmp > x:
                x = tmp
            tmp = min(row)
            if tmp < y:
                y = tmp
        return [y, x]
    else:
        all_rgb_codes = img.reshape(-1, img.shape[-1])
        return np.unique(all_rgb_codes, axis=0)


def draw_keypoints(img, keypoints, color=(0, 0, 255)):
    for kp in keypoints:
        x, y = kp.pt
        cv.circle(img, (int(x), int(y)), 20, color, 3)


def binary_img_to_rgb(bin_img):

    r = len(bin_img)
    c = len(bin_img[0])
    rgb_img = np.zeros((r, c, 3), np.uint8)

    for row in range(r):
        for col in range(c):
            for i in range(3):
                rgb_img[row][col][i] = bin_img[row][col]
    return rgb_img


def detect_blobs_one(search_img, paint_img):
    local_img = cv.cvtColor(search_img, cv.COLOR_BGR2GRAY)
    # eliminate noise
    local_img = cv.erode(local_img, np.ones((2, 2)), iterations=10)
    #local_img = cv.dilate(local_img, np.ones((2, 2)), iterations=32, borderType=cv.BORDER_REFLECT_101)
    # make the background white and candies dark
    cv.bitwise_not(local_img, local_img)

    # Setup SimpleBlobDetector parameters.
    params = cv.SimpleBlobDetector_Params()
    # Filter by Circularity
    params.filterByCircularity = True
    # circular is good
    params.minCircularity = 0.7  # a square is 7.85

    # Filter by Convexity
    params.filterByConvexity = True
    # convex is good
    params.minConvexity = 0.5

    # Filter by Inertia
    params.filterByInertia = True
    # high inertia (roundness) is good
    params.minInertiaRatio = 0.5

    # Create a detector with the parameters
    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(local_img)
    if len(keypoints) > 0:
        # sort by size, with the highly constraining circularity parameters, the candies will be preferred,
        # after sorting, the median and average should agree
        keypoints.sort(key=lambda x: x.size)
        # identify the median size of the candies
        candy_size = keypoints[len(keypoints)//2].size**2
        print(keypoints)

        # candy size identified repeat analysis with size restriction
        params.filterByArea = True
        params.minArea = candy_size - candy_size / 2
        params.maxArea = candy_size + candy_size / 2

        # relax other constraints
        params.minCircularity = 0.4
        params.minConvexity = 0.4
        params.minInertiaRatio = 0.4
        # color has been inverted - dilate now erodes
        local_img = cv.dilate(local_img, np.array([[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]], np.uint8),
                              iterations=3)
        #local_img = cv.dilate(local_img, np.array([[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]], np.uint8), iterations=4, borderType=cv.BORDER_REFLECT_101)

        detector = cv.SimpleBlobDetector_create(params)
        keypoints = detector.detect(local_img)
        print(keypoints)

        # Draw detected blobs as red circles.
        draw_keypoints(local_img, keypoints, (0, 0, 255))
        draw_keypoints(paint_img, keypoints, (0, 0, 255))
        cv.imshow("gray img", local_img)


def remove_background(img):
    # eliminate very large blocks of same color
    pass


def start():
    img = cv.imread("imagesWOvideo/four.jpg", cv.IMREAD_COLOR)

    # find locations of the m&ms
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    # simplify each channel to 2 values, this becomes 8 combinations once channels are recombined
    k = 2
    h = simplify_img(h, k, 1)
    s = simplify_img(s, k, 1)
    v = simplify_img(v, k, 1)
    hsv_ch = [h, s, v]

    # convert back to BGR, colors become distorted
    hsv = cv.cvtColor(cv.merge(hsv_ch), cv.COLOR_HSV2BGR)
    #detect_blobs(hsv, img)
    detect_blobs_one(hsv, img)
    cv.imshow("image", hsv)
    cv.imshow("Original Image", img)

    # _, thresh1 = cv.threshold(diff, 30, 255, cv.THRESH_BINARY)

    cv.waitKey(0)
    cv.destroyAllWindows()


start()
