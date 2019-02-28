import numpy as np
import cv2 as cv

candy_size = 0
cv.namedWindow("dials")
pre_scalars = np.zeros(3)
pst_scalars = np.zeros(3)


def change_slider_preCir(val):
    pre_scalars[0] = val/100
def change_slider_preCon(val):
    pre_scalars[1] = val/100
def change_slider_preInr(val):
    pre_scalars[2] = val/100
def change_slider_pstCir(val):
    pst_scalars[0] = val/100
def change_slider_pstCon(val):
    pst_scalars[1] = val/100
def change_slider_pstInr(val):
    pst_scalars[2] = val/100


cv.createTrackbar("Pre Min Cir", "dials", 0, 100, change_slider_preCir)
cv.createTrackbar("Pre Min Con", "dials", 0, 100, change_slider_preCon)
cv.createTrackbar("Pre Min Inr", "dials", 0, 100, change_slider_preInr)
cv.createTrackbar("Pst Min Cir", "dials", 0, 100, change_slider_pstCir)
cv.createTrackbar("Pst Min Con", "dials", 0, 100, change_slider_pstCon)
cv.createTrackbar("Pst Min Inr", "dials", 0, 100, change_slider_pstInr)


def simplify_img(img, channels, depth=3):
    """
    Reduces the number of unique colors in the given image
    :param img: the image to process (non-destructive)
    :param channels: the number of unique colors to keep
    :param depth: how many color channels the result should have
    :return: the resultant image
    """
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
    """
    Draws a circle of radius 20 centered on each given keypoint
    :param img: image to plot the keypoints on
    :param keypoints: keypoints for plotting
    :param color: color of the circles
    :return: None
    """
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


def detect_blobs_one(search_img, pre_filter_cci, post_filter_cci):
    global candy_size
    # eliminate noise
    if len(search_img.shape) > 2 and search_img.shape[2] == 3:
        local_img = cv.cvtColor(search_img, cv.COLOR_BGR2GRAY)
    else:
        local_img = search_img.__copy__()
    local_img = cv.dilate(local_img, np.array([[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]], np.uint8),
                          iterations=3)

    # Setup SimpleBlobDetector parameters.
    params = cv.SimpleBlobDetector_Params()
    # Filter by Circularity
    params.filterByCircularity = True
    # circular is good
    params.minCircularity = pre_filter_cci[0]  # a square is 7.85

    # Filter by Convexity
    params.filterByConvexity = True
    # convex is good
    params.minConvexity = pre_filter_cci[1]

    # Filter by Inertia
    params.filterByInertia = True
    # high inertia (roundness) is good
    params.minInertiaRatio = pre_filter_cci[2]

    # Create a detector with the parameters
    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(local_img)
    if len(keypoints) > 0:
        # sort by size, with the highly constraining circularity parameters, the candies will be preferred,
        # after sorting, the median and average should agree
        # TODO replace median with a mode
        keypoints.sort(key=lambda x: x.size)
        # identify the median size of the candies
        candy_size = keypoints[len(keypoints)//2].size**2

        # candy size identified repeat analysis with size restriction
        params.filterByArea = True
        params.minArea = candy_size - candy_size / 1.2
        params.maxArea = candy_size + candy_size / 1.2

        # relax other constraints
        params.minCircularity = post_filter_cci[0]
        params.minConvexity = post_filter_cci[1]
        params.minInertiaRatio = post_filter_cci[2]
        # color has been inverted - dilate now erodes
        local_img = cv.dilate(local_img, np.array([[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]], np.uint8),
                              iterations=3)
        #local_img = cv.dilate(local_img, np.array([[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]], np.uint8), iterations=4, borderType=cv.BORDER_REFLECT_101)

        detector = cv.SimpleBlobDetector_create(params)
        keypoints = detector.detect(local_img)

    return keypoints


def outline_img(img):
    # eliminate very large blocks of same color
    local_img = cv.Laplacian(img, 5)
    # local_img = cv.dilate(local_img, np.array([[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]], np.uint8),
    #                       iterations=1)
    local_img = cv.cvtColor(local_img, cv.COLOR_BGR2GRAY)
    _, local_img = cv.threshold(local_img, 1, 255, cv.THRESH_BINARY)
    local_img = np.array(local_img, np.uint8)

    # cv.imshow("outline", local_img)
    return local_img


def same_rgb(rgb1, rgb2):
    """
    Returns true if both given RGB values are the same
    :param rgb1:
    :param rgb2:
    :return: True/False
    """
    return rgb1[0] == rgb2[0] and rgb1[1] == rgb2[1] and rgb1[2] == rgb2[2]


def remove_background(img):
    """
    Sets most of the background to white by flood filling the perimeter
    :param img: to work on (non-destructive)
    :return: img with no background
    """
    local_img = img.__copy__()

    # get all color transitions along the perimeter of the image
    last_pix = 255*np.ones([3], np.uint8)
    boundaries = []
    h, w = img.shape[:2]

    # build perimeter path
    s0 = np.zeros([w, 2], np.uint16)
    s1 = np.zeros([h, 2], np.uint16)
    s2 = np.zeros([w, 2], np.uint16)
    s3 = np.zeros([h, 2], np.uint16)

    for i in range(w):
        s0[i][1] = i
        s2[i][0] = h - 1
        s2[i][1] = i
    for i in range(h):
        s3[i][0] = i
        s1[i][1] = w - 1
        s1[i][0] = i

    perimeter = np.append(s0, s1, axis=0)
    perimeter = np.append(perimeter, s2, axis=0)
    perimeter = np.append(perimeter, s3, axis=0)

    for row, col in perimeter:
        test = same_rgb(img[row][col], last_pix)
        if test is not None and not test:
            boundaries.append([row, col])
            last_pix = img[row][col]

    # flood fill w/ white on each section along the perimeter
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for point in boundaries:
        cv.floodFill(local_img, mask, (point[1], point[0]), (255, 255, 255))

    return local_img


def draw_blob(img, verts):
    """
    Draws a line around a area given by the list of vertexes, given in the weird OpenCV list of [[x, y]]
    :param img: the image to draw the blob on
    :param verts: the bounds of the blob given in the order around the blob
    :return: None
    """
    for i in range(len(verts) - 1):
        v1 = tuple(verts[i][0])
        v2 = tuple(verts[i+1][0])
        cv.line(img, v1, v2, (255, 0, 0), 10)
    cv.line(img, tuple(verts[-1][0]), tuple(verts[0][0]), (255, 0, 0), 10)


def merge_keypoints(first_set, second_set, area):
    """
    Merges the given numpy arrays of keypoints and dose not add keypoints that overlap
    :param first_set:
    :param second_set:
    :param area: area of a candy
    :return: numpy array of non-duplicate keypoints
    """
    candy_len = np.sqrt(area)
    local_set = [x for x in first_set]
    for point in second_set:
        append = True
        point = np.array(point.pt)
        for com_point in first_set:
            com_point = np.array(com_point.pt)
            if (point - com_point <= candy_len).all():
                append = False
                break
        if append:
            np.append(local_set, point)
    return local_set


def threshold_img(img, min_scalars, max_scalars):
    """
    Thresholds the given image according to the given ranges, also removes noise and inverts the binary image
    :param img: the image to process (non-destructive)
    :param min_scalars:
    :param max_scalars:
    :return: processed image
    """
    local_img = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT)
    tracking_base = cv.inRange(local_img, min_scalars, max_scalars)
    # remove image noise
    tracking = cv.erode(tracking_base, np.ones((6, 6)))
    tracking = cv.dilate(tracking, np.ones((6, 6)))
    # invert
    tracking = cv.bitwise_not(tracking, tracking)
    return tracking


def detect_candy(file, window_name):
    global candy_size, pre_scalars, pst_scalars
    img = cv.imread(file, cv.IMREAD_COLOR)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)

    # find locations of the m&ms (not brown)
    min_hsv = np.array([0, 72, 150], np.uint8)
    max_hsv = np.array([180, 255, 255], np.uint8)
    tracking_base = threshold_img(hsv, min_hsv, max_hsv)

    # find locations of the brown m&ms
    min_hls = np.array([23, 0, 0], np.uint8)
    max_hls = np.array([255, 255, 255], np.uint8)
    tracking_brown = threshold_img(hls, min_hls, max_hls)

    # get keypoints
    keypoints_base = detect_blobs_one(tracking_base, pre_scalars, pst_scalars)
    keypoints_brown = detect_blobs_one(tracking_brown, pre_scalars, pst_scalars)
    # print("KP B:", keypoints_brown)
    # print("KP NB:", keypoints_base)
    keypoints = merge_keypoints(keypoints_base, keypoints_brown, candy_size)
    # print("KP T:", keypoints)

    # cv.imshow("tracking base " + window_name, tracking_base)
    # cv.imshow("tracking brown " + window_name, tracking_brown)
    # Draw detected blobs as red circles.
    draw_keypoints(img, keypoints, (0, 0, 255))

    cv.imshow(window_name, img)


while True:
    detect_candy("imagesWOvideo/one.jpg", "One")
    # detect_candy("imagesWOvideo/two.jpg", "Two")
    # detect_candy("imagesWOvideo/three.jpg", "Three")
    detect_candy("imagesWOvideo/four.jpg", "Four")
    # detect_candy("imagesWOvideo/001.jpg", "001")
    # detect_candy("imagesWOvideo/002.jpg", "002")

    k = cv.waitKey(1)
    # this is the "esc" key
    if k == 27:
        break

cv.destroyAllWindows()
