import cv2.cv2 as cv2
import numpy as np
from utils.ImageColorCorrection import ImageColorCorrection
import matplotlib.pyplot as plt
from utils.evaluate_result import evaluate



if __name__ == '__main__':
    # image = cv2.imread(r"data\image\Camerasnap6.jpg")[..., ::-1] / 255.
    # image_wb = cv2.imread(r"data\image\Camerasnap9.jpg")[..., ::-1] / 255.
    # image = cv2.imread(r"./data/mindvision/mvmv_2300.PNG", -1) / 65535.
    # image_wb = cv2.imread(r"./data/mindvision/mvmv_2300.PNG", -1) / 65535.

    image_wb = np.fromfile(r'image\mv1_scene\cc_24.RAW', dtype=np.uint8).reshape(2048, -1)
    image_wb = cv2.cvtColor(image_wb, cv2.COLOR_BayerRG2BGR)[..., ::-1] / 255.

    image = image_wb.copy()
    # image = np.fromfile(r'image\mv1_scene\image_53.RAW', dtype=np.uint8).reshape(2048, -1)
    # image = cv2.cvtColor(image, cv2.COLOR_BayerRG2BGR)[..., ::-1] / 255.

    # plt.figure()
    # plt.imshow(image)
    # plt.show()
    

    cct_ccm_dict = np.load(r"data_calib\mv1_xrite_3x4.npy", allow_pickle=True).item()
    image_color_correction = ImageColorCorrection(cct_ccm_dict, "linear")
    image_color_correction.setMethod("cc")
    image_color_correction.doWhiteBalance(wb_image=image_wb)
    corrected_image, corrected_image_wb = image_color_correction.correctImage(image, "linear")

    ideal_lab_3 = np.float32(np.loadtxt("./data/real_lab_xrite.csv", delimiter=','))  # from 3nh
    deltaC, deltaE00, img_with_gt, deltaE76, deltaC76 = evaluate(corrected_image, ideal_lab_3, 'linear', 'deltaC')
    img_with_gt = np.clip(img_with_gt, 0, 1)
    print('deltaC00, deltaE00 - mean: ', deltaC.mean(), deltaE00.mean(), deltaC76.mean(), deltaE76.mean())
    print('deltaC00, deltaE00 - max: ', deltaC.max(), deltaE00.max(), deltaC76.max(), deltaE76.max())


    # cv2.imwrite('img_corrected_gamma.png',corrected_image[...,::-1]** (1/2.2) * 255.)
    # cv2.imwrite('img_corrected.png',corrected_image[...,::-1] * 255.)

    # cv2.imwrite('img_corrected_wb.png',corrected_image_wb[...,::-1] * 255.)


    # plt.figure()
    # plt.imshow(image)
    # plt.figure()
    # plt.imshow(corrected_image ** (1/2.2))
    # plt.show()