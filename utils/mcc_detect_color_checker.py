import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt



def transform_points_forward(_T, X):
    p = np.array([[X[0]], [X[1]], [1]])
    # xt = _T * p
    xt = np.dot(_T, p)
    return np.array([xt[0, 0]/xt[2, 0], xt[1,0]/xt[2,0]])

def CGetCheckerCentroid(checker):
    cellchart = np.array([[1.50, 1.50], [4.25, 1.50], [7.00, 1.50], [9.75, 1.50], [12.50, 1.50], [15.25, 1.50],
    [1.50, 4.25], [4.25, 4.25], [7.00, 4.25], [9.75, 4.25], [12.50, 4.25], [15.25, 4.25], [1.50, 7.00],
    [4.25, 7.00], [7.00, 7.00], [9.75, 7.00], [12.50, 7.00], [15.25, 7.00], [1.50, 9.75], [4.25, 9.75],
    [7.00, 9.75], [9.75, 9.75], [12.50, 9.75], [15.25, 9.75]])
    center = checker.getCenter()
    box = checker.getBox()
    size = np.array([4, 6])
    boxsize = np.array([11.25, 16.75])
    fbox = np.array([[0.00, 0.00], [16.75, 0.00], [16.75, 11.25], [0.00, 11.25]])
    # print(fbox, box)
    ccT = cv2.getPerspectiveTransform(np.float32(fbox), np.float32(box))
    sorted_centroid = []
    for i in range(24):
         Xt = transform_points_forward(ccT, cellchart[i])
         sorted_centroid.append(Xt)
    return sorted_centroid

def detect_color_checker(image):
    # image = image / 255
    image = image ** (1 / 2.2) * 255
    image = np.uint8(image)
    detector = cv2.mcc.CCheckerDetector_create()
    detector.process(image, cv2.mcc.MCC24, 1, True)

    checkers = detector.getListColorChecker()
    print('len(checkers): ',len(checkers))
    checker = checkers[0]
    cdraw = cv2.mcc.CCheckerDraw_create(checker)
    img_draw = image.copy()
    cdraw.draw(img_draw)

    chartsRGB = checker.getChartsRGB()
    width, height = chartsRGB.shape[:2]
    roi = chartsRGB[0:width, 1]
    rows = int(roi.shape[:1][0])
    src = chartsRGB[:, 1].copy().reshape(int(rows / 3), 1, 3)

    sorted_centroid = CGetCheckerCentroid(checker)
    marker_image = image.copy()
    # print(sorted_centroid)
    for num, centroid in enumerate(sorted_centroid):
        cv2.putText(marker_image, str(num), np.int32(centroid), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)

    sorted_centroid = np.array(sorted_centroid)

    return sorted_centroid, sorted_centroid, marker_image



if __name__ == '__main__':
    image = cv2.imread(r"data\mindvision\exposure30.jpg")
    _, _, image = detect_color_checker(image[:, :, ::-1].copy()/255.)
    plt.figure()
    plt.imshow(image)
    plt.show()

