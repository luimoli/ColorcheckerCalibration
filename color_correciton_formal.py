import cv2.cv2 as cv2
import numpy as np
from utils.ImageColorCorrection import ImageColorCorrection
import matplotlib.pyplot as plt
from utils.evaluate_result import evaluate

if __name__ == '__main__':
    # image = cv2.imread(r"data\image\Camerasnap6.jpg")[..., ::-1] / 255.
    # image_wb = cv2.imread(r"data\image\Camerasnap9.jpg")[..., ::-1] / 255.
    image = cv2.imread(r"./data/mindvision/mv_2300.PNG")[..., ::-1] / 255.
    image_wb = cv2.imread(r"./data/mindvision/mv_2300.PNG")[..., ::-1] / 255.

    cct_ccm_dict = np.load("./data/calibration_3nh.npy", allow_pickle=True).item()
    image_color_correction = ImageColorCorrection(cct_ccm_dict, "linear")
    image_color_correction.setMethod("multiple_light")
    image_color_correction.doWhiteBalance(wb_image=image_wb)
    corrected_image = image_color_correction.correctImage(image, "linear")

    # ideal_lab_3 = np.float32(np.loadtxt("./data/real_lab_d50_3ns.csv", delimiter=','))  # from 3nh
    # deltaC, deltaE00, img_with_gt = evaluate(corrected_image, ideal_lab_3, 'linear', 'deltaC')
    # img_with_gt = np.clip(img_with_gt, 0, 1)
    # print('deltaC00, deltaE00: ', deltaC.mean(), deltaE00.mean())

    plt.figure()
    plt.imshow(image)
    plt.figure()
    plt.imshow(corrected_image ** (1/2.2))
    plt.show()