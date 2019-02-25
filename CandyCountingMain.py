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


def detect_blobs(search_img, paint_img):
    # detect blobs of correct size
    # get all unique colors in reduced detection image
    unique_colors = get_unique_colors(search_img)
    print(unique_colors)
    # threshold each unique color
    # Setup SimpleBlobDetector parameters.
    params = cv.SimpleBlobDetector_Params()
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 500
    params.maxArea = 1200
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.35

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.1

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    i = 0
    master_contour_list = []
    for color in unique_colors:
        tmp = cv.inRange(search_img, color, color)
        tmp_inv = tmp.__copy__()
        cv.bitwise_not(tmp, tmp_inv)

        # Create a detector with the parameters
        detector = cv.SimpleBlobDetector_create(params)

        print(len(tmp), len(tmp[0]))
        tmp = binary_img_to_rgb(tmp)
        tmp = cv.erode(tmp, np.ones((3, 3)))
        tmp = cv.dilate(tmp, np.ones((3, 3)))

        keypoints_std = detector.detect(tmp)
        keypoints_inv = detector.detect(tmp_inv)
        for keypoints in [keypoints_std, keypoints_inv]:
            print(len(keypoints))
            # Draw detected blobs as red circles.
            draw_keypoints(tmp, keypoints, (0, 0, 255))
            draw_keypoints(paint_img, keypoints, (0, 0, 255))
            draw_keypoints(search_img, keypoints, (0, 0, 255))

        #cv.imshow("tmp " + str(i), tmp)
        #i += 1


def start():
    img = cv.imread("imagesWOvideo/one.jpg", cv.IMREAD_COLOR)
    cv.imshow("Original Image", img)

    # find locations of the m&ms
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    # simplify each channel to 2 values, this becomes 8 combinations once channels are recombined
    k = 2
    h = simplify_img(h, k, 1)
    s = simplify_img(s, k, 1)
    v = simplify_img(v, k, 1)
    hsv_ch = [h, s, v]

    # # threshold each channel
    # for i in range(len(hsv_ch)):
    #     colors = get_unique_colors(hsv_ch[i], True)
    #     print(colors)
    #     _, hsv_ch[i] = cv.threshold(hsv_ch[i], colors[0] + 1, 255, cv.THRESH_BINARY)
    #     cv.imshow("threshold" + str(i), hsv_ch[i])

    # cv.imshow("h", h)
    # cv.imshow("s", s)
    # cv.imshow("v", v)

    # convert back to BGR, colors become distorted
    hsv = cv.cvtColor(cv.merge(hsv_ch), cv.COLOR_HSV2BGR)
    detect_blobs(hsv, img)
    #detect_blobs_rgb(hsv, img)
    cv.imshow("image", hsv)

    # _, thresh1 = cv.threshold(diff, 30, 255, cv.THRESH_BINARY)

    cv.waitKey(0)
    cv.destroyAllWindows()


start()
