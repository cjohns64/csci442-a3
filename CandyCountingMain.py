import numpy as np
import cv2 as cv

candy_size = 0
size_thresh = np.array([1.0, 1.0])
invert = True
count_orange = 0
count_yellow = 0
count_red = 0
count_blue = 0
count_green = 0
count_brown = 0

orange_range = np.zeros([3], np.uint8)
yellow_range = np.zeros([3], np.uint8)
red_range = np.zeros([3], np.uint8)
blue_range = np.zeros([3], np.uint8)
green_range = np.zeros([3], np.uint8)
brown_range = np.zeros([3], np.uint8)


def set_color_ranges():
    global orange_range, yellow_range, red_range, blue_range, green_range, brown_range
    """
    Average color values calculated from a large sample of BGR values
    Blue
    253.16, 179.54, 2.78
    
    Brown
    92.11, 84.73, 83.22
    
    Yellow
    29.0, 237.96, 247.2
    
    Orange
    61.087, 118.96, 244.83
    
    Red
    105.2, 89.08, 208.44
    
    Green
    128.42, 220.56, 6.94
    """
    # set orange
    orange_range[0], orange_range[1], orange_range[2] = 61.087, 118.96, 244.83

    # set red
    red_range[0], red_range[1], red_range[2] = 105.2, 89.08, 208.44

    # set green
    green_range[0], green_range[1], green_range[2] = 128.42, 220.56, 6.94

    # set yellow
    yellow_range[0], yellow_range[1], yellow_range[2] = 29.0, 237.96, 247.2

    # set blue
    blue_range[0], blue_range[1], blue_range[2] = 253.16, 179.54, 2.78

    # set brown
    brown_range[0], brown_range[1], brown_range[2] = 92.11, 84.73, 83.22


def draw_keypoints(img, keypoints, color=(0, 0, 255)):
    """
    Draws a circle of radius 20 centered on each given keypoint
    :param img: image to plot the keypoints on
    :param keypoints: keypoints for plotting
    :param color: color of the circles
    :return: None
    """
    # test if keypoints is a list of some kind
    try:
        len(keypoints)
    except TypeError:
        keypoints = [keypoints]

    for kp in keypoints:
        x, y = kp.pt
        cv.circle(img, (int(x), int(y)), 12, color, 3)


def detect_blobs_one(search_img):
    global candy_size, invert, size_thresh
    # eliminate noise
    if len(search_img.shape) > 2 and search_img.shape[2] == 3:
        local_img = cv.cvtColor(search_img, cv.COLOR_BGR2GRAY)
    else:
        local_img = search_img.__copy__()
    if invert:
        local_img = cv.dilate(local_img, np.ones((6, 6)), iterations=2)
    else:
        local_img = cv.erode(local_img, np.ones((6, 6)), iterations=2)

    # setup SimpleBlobDetector parameters.
    params = cv.SimpleBlobDetector_Params()
    # filter by circularity
    params.filterByCircularity = True
    params.minCircularity = 0.2  # a square is 7.85
    # filter by convexity
    params.filterByConvexity = True
    params.minConvexity = 0.2
    # filter by inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.2

    # Create a detector with the parameters
    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(local_img)
    if len(keypoints) > 0:
        # sort by size, with the highly constraining circularity parameters, the candies will be preferred,
        # after sorting, the median and average should agree
        keypoints.sort(key=lambda x: x.size)
        # identify the median size of the candies
        candy_size = keypoints[len(keypoints)//2].size**2

        # candy size identified repeat analysis with size restriction
        params.filterByArea = False
        # params.minArea = candy_size - candy_size / size_thresh[0]
        # params.maxArea = candy_size + candy_size / size_thresh[1]

        params.minCircularity = 0
        params.minConvexity = 0
        params.minInertiaRatio = 0
        detector = cv.SimpleBlobDetector_create(params)
        keypoints = detector.detect(local_img)

    return keypoints


def merge_keypoints(first_set, second_set):
    """
    Merges the given numpy arrays of keypoints and dose not add keypoints that overlap
    :param first_set:
    :param second_set:
    :return: numpy array of non-duplicate keypoints
    """
    local_set = np.append(first_set, second_set)
    return local_set


def threshold_img(img, min_scalars, max_scalars):
    """
    Thresholds the given image according to the given ranges, also removes noise and inverts the binary image
    :param img: the image to process (non-destructive)
    :param min_scalars:
    :param max_scalars:
    :return: processed image
    """
    global invert
    local_img = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT)
    tracking_base = cv.inRange(local_img, min_scalars, max_scalars)
    # remove image noise
    tracking = cv.erode(tracking_base, np.ones((6, 6)), iterations=1)
    tracking = cv.dilate(tracking, np.ones((6, 6)), iterations=1)
    # invert
    if invert:
        tracking = cv.bitwise_not(tracking, tracking)
    return tracking


def coord_out_of_bounds(img, coord):
    return (coord[0] >= len(img) or coord[1] >= len(img[0]) or
            coord[0] < 0 or coord[1] < 0)


def fit_to_color(bgr, img, keypt):
    """
    Identifies colors based off of known spread of values from a sample average
    increments color count and plots bounding circle
    :param bgr:
    :param img:
    :param keypt: keypoint of interest
    :return:
    """
    global orange_range, yellow_range, red_range, blue_range, green_range, brown_range
    global count_orange, count_yellow, count_red, count_blue, count_green, count_brown

    # blue
    if (np.abs(bgr - blue_range) < np.array([12, 115, 20])).all():
        count_blue += 1
        draw_keypoints(img, keypt, (255, 0, 0))
    # yellow
    elif (np.abs(bgr - yellow_range) < np.array([137, 63, 36])).all():
        count_yellow += 1
        draw_keypoints(img, keypt, (0, 250, 250))
    # orange
    elif (np.abs(bgr - orange_range) < np.array([153, 119, 33])).all():
        count_orange += 1
        draw_keypoints(img, keypt, (0, 140, 250))
    # green
    elif (np.abs(bgr - green_range) < np.array([197, 104, 120])).all():
        count_green += 1
        draw_keypoints(img, keypt, (0, 250, 0))
    # red
    elif (np.abs(bgr - red_range) < np.array([178, 148, 105])).all():
        count_red += 1
        draw_keypoints(img, keypt, (0, 0, 250))
    # brown
    elif (np.abs(bgr - brown_range) < np.array([229, 192, 156])).all():
        count_brown += 1
        draw_keypoints(img, keypt, (120, 140, 160))
    else:
        draw_keypoints(img, keypt, (255, 255, 255))


def count_colors(img, keypoints):
    # count identified objects
    n = 0
    size = 1
    for kp in keypoints:
        col, row = kp.pt

        avg_color = np.zeros(3)
        i = 0
        for x in range(2*size):
            for y in range(2*size):
                eval_row = int(row - size//2 + x)
                eval_col = int(col - size//2 + y)

                if not coord_out_of_bounds(img, [eval_row, eval_col]):
                    # accumulate average
                    avg_color += img[eval_row][eval_col]
                    i += 1
        if i > 0:
            n += 1
            avg_color /= i
            fit_to_color(avg_color, img, kp)
        else:
            print(int(row), int(col))
    # print results
    print(n)
    text = "yellow: {},blue: {},green: {},orange: {},red: {},brown: {}".format(
        count_yellow, count_blue, count_green, count_orange, count_red, count_brown)
    text_arr = text.split(",")

    for i in range(len(text_arr)):
        cv.putText(img, text_arr[i], (10, 20+22*i), cv.FONT_HERSHEY_PLAIN, 1.5, (240, 240, 240), thickness=2, lineType=5)


def detect_candy(file, window_name):
    global candy_size, pre_scalars, pst_scalars
    candy_size = 0
    img = cv.imread(file, cv.IMREAD_COLOR)
    img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    blue_green = img.__copy__()
    yellow_orange = img.__copy__()

    # find locations of the m&ms (not brown)
    min_BG = np.array([0, 137, 0], np.uint8)
    max_BG = np.array([255, 255, 146], np.uint8)
    tracking_BG = threshold_img(blue_green, min_BG, max_BG)
    #tracking_BG = cv.dilate(tracking_BG, np.ones((2, 2)), iterations=2)

    # find locations of the brown m&ms
    min_YO = np.array([0, 0, 205], np.uint8)
    max_YO = np.array([255, 255, 255], np.uint8)
    tracking_YO = threshold_img(yellow_orange, min_YO, max_YO)
    tracking_YO = cv.dilate(tracking_YO, np.ones((2, 2)), iterations=1)
    cv.imshow("T YO", tracking_YO)
    cv.imshow("T BG", tracking_BG)

    # get keypoints
    keypoints_BG = detect_blobs_one(tracking_BG)
    keypoints_YO = detect_blobs_one(tracking_YO)
    keypoints = merge_keypoints(keypoints_BG, keypoints_YO)

    # draw_keypoints(img, keypoints_BG)
    # draw_keypoints(img, keypoints_YO)
    # count and print the results
    count_colors(img, keypoints)

    cv.imshow(window_name, img)


set_color_ranges()
detect_candy("imagesWOvideo/candyBigSmallerTiny.jpg", "All")

cv.waitKey(0)
cv.destroyAllWindows()
