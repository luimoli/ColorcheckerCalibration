import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
def is_square(contour, tolerance=0.005):
    return cv2.matchShapes(contour, np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
                           cv2.CONTOURS_MATCH_I2, 0.0) < tolerance


def contour_centroid(contour):
    """
    Returns the centroid of given contour.

    Parameters
    ----------
    contour : array_like
        Contour to return the centroid of.

    Returns
    -------
    tuple
        Contour centroid.

    Notes
    -----
    -   A :class:`tuple` class is returned instead of a :class:`ndarray` class
        for convenience with *OpenCV*.

    Examples
    --------
    >>> contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> contour_centroid(contour)
    (0.5, 0.5)
    """

    moments = cv2.moments(contour)
    centroid = np.array(
        [moments['m10'] / moments['m00'], moments['m01'] / moments['m00']])

    return np.array([centroid[0], centroid[1]])


def scale_contour(contour, factor):
    """
    Scales given contour by given scale factor.

    Parameters
    ----------
    contour : array_like
        Contour to scale.
    factor : numeric
        Scale factor.

    Returns
    -------
    ndarray
        Scaled contour.

    Examples
    --------
    >>> contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> scale_contour(contour, 2)
    array([[ 0.,  0.],
           [ 2.,  0.],
           [ 2.,  2.],
           [ 0.,  2.]])
    """

    centroid = (contour_centroid(contour))
    scaled_contour = ((contour) - centroid) * factor + centroid

    return scaled_contour


def find_contour(image):
    block_size = int(1440 * 0.015)
    image = np.uint8(image)
    image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    image_s = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, block_size, 3)

    kernel = np.ones((3, 3), np.uint8)
    image_c = cv2.erode(image_s, kernel, iterations=1)
    image_c = cv2.dilate(image_c, kernel, iterations=1)
    # plt.figure()
    # plt.imshow(image_c)
    # plt.show()

    contours, _hierarchy = cv2.findContours(image_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    swatches = []
    for contour in contours:
        curve = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if 10000 < cv2.contourArea(curve) < 400000 and is_square(curve):
            # print(cv2.contourArea(curve))
            swatches.append((cv2.boxPoints(cv2.minAreaRect(curve))))
    return swatches


def find_centroid(swatches, image):
    # swatches = find_contour(image_r) + find_contour(image_g) + find_contour(image_b)
    clusters = np.zeros(image.shape, dtype=np.uint8)
    centroid_list, area_list = [], []
    threshold = 50
    for i in swatches:
        flag = True
        centroid = contour_centroid(i)
        for j in range(len(centroid_list)):
            if np.abs(centroid - centroid_list[j]).sum() < threshold:
                flag = False
                break
        if flag:
            cv2.circle(clusters, np.int32(centroid), 20, (255, 255, 255))
            cv2.line(clusters, np.int32(i[0]), np.int32(i[1]), (255, 255, 255))
            cv2.line(clusters, np.int32(i[1]), np.int32(i[2]), (255, 255, 255))
            cv2.line(clusters, np.int32(i[2]), np.int32(i[3]), (255, 255, 255))
            cv2.line(clusters, np.int32(i[3]), np.int32(i[0]), (255, 255, 255))
            centroid_list.append(np.int32(centroid))
            area_list.append(abs(i[1]-i[0]) * abs(i[2]-i[1]))

    return clusters, np.array(centroid_list), np.array(area_list)


def detect_color_checker(image):
    image = np.uint8(image * 255.)
    image2 = np.copy(image)
    image2 = np.uint8((image2 / 255) ** (1 / 2.2) * 255)
    image_r, image_g, image_b = image[..., 0], image[..., 1], image[..., 2]
    image_r2, image_g2, image_b2 = image2[..., 0], image2[..., 1], image2[..., 2]
    x1 = np.uint8(np.abs(np.int32(image2[..., 0]) - np.int32(image2[..., 1])))
    x2 = np.uint8(np.abs(np.int32(image2[..., 1]) - np.int32(image2[..., 2])))
    x3 = np.uint8(np.abs(np.int32(image2[..., 2]) - np.int32(image2[..., 0])))
    x4 = np.uint8(np.clip(np.int32(image[..., 2]) * np.int32(image[..., 0]), 0, 255))
    x5 = np.uint8(np.clip(np.int32(image[..., 0]) * np.int32(image[..., 1]), 0, 255))
    x6 = np.uint8(np.clip(np.int32(image[..., 1]) * np.int32(image[..., 2]), 0, 255))

    swatches = find_contour(image_r) + find_contour(image_g) + find_contour(image_b) + \
               find_contour(image_r2) + find_contour(image_g2) + find_contour(image_b2) + \
               find_contour(x1) + find_contour(x2) + find_contour(x3) \
               + find_contour(x4) + find_contour(x5) + find_contour(x6)
    clusters, centroid_array, area_array = find_centroid(swatches, image)
    white_gray = 0
    for centroid in centroid_array:
        mean_value = image[centroid[1] - 50:centroid[1] + 50, centroid[0] - 50:centroid[0] + 50].mean(axis=(0, 1))
        if mean_value.mean() > white_gray:
            white_centroid = centroid
            white_gray = mean_value.mean()

    distance = np.mean(np.abs(white_centroid - centroid_array), axis=1)
    distance_index = np.argsort(distance)[1:3]

    rgb_mean = image[white_centroid[1] - 50:white_centroid[1] + 50,
               white_centroid[0] - 50:white_centroid[0] + 50].mean(axis=(0, 1))

    rgb_gain = rgb_mean / rgb_mean.max()

    # r_gain = rgb_mean[1] / rgb_mean[0]
    # b_gain = rgb_mean[1] / rgb_mean[2]
    image = image / 255.

    image[..., 0] *= rgb_gain[0]
    image[..., 1] *= rgb_gain[1]
    image[..., 2] *= rgb_gain[2]

    centroid1, centroid2 = centroid_array[distance_index]
    mean_value1 = image[centroid1[1] - 10:centroid1[1] + 10, centroid1[0] - 10:centroid1[0] + 10].mean(axis=(0, 1))
    mean_value2 = image[centroid2[1] - 10:centroid2[1] + 10, centroid2[0] - 10:centroid2[0] + 10].mean(axis=(0, 1))
    # print(centroid1, centroid2)
    # print(mean_value1, mean_value2)
    # plt.figure()
    # plt.imshow(image)
    # plt.show()
    # if (mean_value1[1] / mean_value1[0] > mean_value2[1] / mean_value2[0]):
    #     blue_centroid, gray_centroid_2 = centroid1, centroid2
    # else:
    #     blue_centroid, gray_centroid_2 = centroid2, centroid1

    # if (mean_value1[1] * mean_value1[0] < mean_value2[1] * mean_value2[0]):
    #     blue_centroid, gray_centroid_2 = centroid1, centroid2
    # else:
    #     blue_centroid, gray_centroid_2 = centroid2, centroid1

    mean_value1[mean_value1 == 0] = 0.0001
    mean_value2[mean_value2 == 0] = 0.0001
    if (mean_value1[1] * mean_value1[0] * mean_value1[2] < mean_value2[2] * mean_value2[1] * mean_value2[0]):
        blue_centroid, gray_centroid_2 = centroid1, centroid2
    else:
        blue_centroid, gray_centroid_2 = centroid2, centroid1

    offset_col = gray_centroid_2 - white_centroid
    offset_row = blue_centroid - white_centroid

    # print("gray_centroid_2:", gray_centroid_2)
    # print("white_centroid:", white_centroid)

    sorted_centroid = np.empty((24, 2))
    sorted_centroid[18] = white_centroid
    sorted_centroid[19] = gray_centroid_2
    sorted_centroid[12] = blue_centroid
    threshold = 10
    for i in range(20, 24):
        candidate_centroid = sorted_centroid[i - 1] + offset_col
        distance = np.mean(np.abs(candidate_centroid - centroid_array), axis=1)
        distance_index = np.argsort(distance)[0]
        min_distance = distance[distance_index]
        if (min_distance <= threshold):
            sorted_centroid[i] = centroid_array[distance_index]
        else:
            sorted_centroid[i] = candidate_centroid

    for i in range(17, -1, -1):
        candidate_centroid = sorted_centroid[i + 6] + offset_row
        distance = np.mean(np.abs(candidate_centroid - centroid_array), axis=1)
        distance_index = np.argsort(distance)[0]
        min_distance = distance[distance_index]
        if (min_distance <= threshold):
            sorted_centroid[i] = centroid_array[distance_index]
        else:
            sorted_centroid[i] = candidate_centroid

    marker_image = np.copy(image)
    for num, centroid in enumerate(sorted_centroid):
        cv2.putText(marker_image, str(num), np.int32(centroid), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)
    return sorted_centroid, clusters, marker_image


