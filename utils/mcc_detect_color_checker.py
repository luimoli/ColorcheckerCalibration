import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

def transform_points_forward(_T, X):
    p = np.array([[X[0]], [X[1]], [1]])
    # xt = _T * p
    xt = np.dot(_T, p)
    return np.array([xt[0, 0]/xt[2, 0], xt[1,0]/xt[2,0]])

def CGetCheckerCentroid(checker):
    cellchart = np.array([[1.50, 1.50], [4.25, 1.50], [7.00,1.50], 
                          [9.75, 1.50], [12.50, 1.50], [15.25, 1.50], 
                          [1.50, 4.25], [4.25, 4.25], [7.00, 4.25], 
                          [9.75, 4.25], [12.50, 4.25], [15.25, 4.25], 
                          [1.50, 7.00], [4.25, 7.00], [7.00, 7.00], 
                          [9.75, 7.00], [12.50, 7.00], [15.25, 7.00], 
                          [1.50, 9.75], [4.25, 9.75], [7.00, 9.75], 
                          [9.75, 9.75], [12.50, 9.75], [15.25, 9.75]])
    center = checker.getCenter()
    box = checker.getBox()
    size = np.array([4, 6])
    boxsize = np.array([11.25, 16.75])
    fbox = np.array([[0.00, 0.00], [16.75, 0.00], [16.75, 11.25], [0.00, 11.25]])
    pixel_distance = ((box[0] - box[1]) ** 2).sum() ** 0.5
    block_pixel = pixel_distance / 6.68
    ccT = cv2.getPerspectiveTransform(np.float32(fbox), np.float32(box))
    sorted_centroid = []
    for i in range(24):
        Xt = transform_points_forward(ccT, cellchart[i])
        sorted_centroid.append(Xt)
    return np.int32(np.array(sorted_centroid)), block_pixel


def calculate_colorchecker_value(image, sorted_centroid, length):
    sorted_centroid2 = sorted_centroid.copy()
    mean_value = np.empty((sorted_centroid.shape[0], 3))
    for i in range(len(sorted_centroid)):
        mean_value[i] = np.mean(image[sorted_centroid2[i, 1] -
                                      length:sorted_centroid2[i, 1] + length,
                                      sorted_centroid2[i, 0] -
                                      length:sorted_centroid2[i, 0] + length],
                                axis=(0, 1))
    return np.float32(mean_value)


def detect_colorchecker(image):
    """detect colorchecker's patches and determine the mean RGB value.
    Args:
        image (arr): channel order: R-G-B | range: [0,1] | colorspace:'linear'
    Returns:
        list: [sorted_centroid: coordinates of color patches' centers.
               length: range of the center area.
               charts_RGB: calculated mean RGB value of color patches using sorted_centroid and length .
               marker_image: mark the detection on image.
               ]
    """
    image_for_detect = np.uint8(255 * image[..., ::-1].copy())
    detector = cv2.mcc.CCheckerDetector_create()
    detector.process(image_for_detect, cv2.mcc.MCC24, 1, True)

    checkers = detector.getListColorChecker()
    checker = checkers[0]
    cdraw = cv2.mcc.CCheckerDraw_create(checker)
    img_draw = image_for_detect.copy()
    cdraw.draw(img_draw)

    sorted_centroid, block_pixel = CGetCheckerCentroid(checker)
    length = np.int32(block_pixel // 4)

    charts_RGB = calculate_colorchecker_value(image, sorted_centroid, length)

    marker_image = image.copy()
    for num, centroid in enumerate(sorted_centroid):
        cv2.putText(marker_image, str(num), np.int32(centroid),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)

    return sorted_centroid, length, charts_RGB, marker_image



if __name__ == '__main__':
    image = cv2.imread(r"D:\Code\CailibrationChecker\data\mindvision\exposure30.jpg")
    _, _, _, image = detect_colorchecker(image[:, :, ::-1].copy()/255.)
    plt.figure()
    plt.imshow(image)
    plt.show()
