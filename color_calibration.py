import os
# from time import CLOCK_MONOTONIC_RAW
from utils.CCM_function2 import ImageColorCorrection
import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import smv_colour
import torch


def generate_calib(image_calib, ccm_space, image_color_space, gt_form, savez_path, ccm_weight=np.ones((24))):
    """This should generate the standalized calibrated CCM.
        'image_calib' is collected under specific illuminant.

    Args:
        image_calib (array): sensor-RGB-IMG
        ccm_space (str): ['srgb','linear']
        image_color_space (str): color-space of the collected image which is used for calibration. ['srgb','linear']
        gt_form (str): decide  ['xrite', 'imatest]
        savez_path (str): save the calibrated CCM and CCT.
    """
    config_minimize = {"method": "minimize", "ccm_space": ccm_space, "gt_form": gt_form, "ccm_weight": ccm_weight}
    icc_minimize = ImageColorCorrection(config_minimize)
    icc_minimize.update_message_from_colorchecker(image_calib)
    image_ccm = icc_minimize.image_correction(image_calib, image_color_space, white_balance=True, illumination_gain=True,
                                                    ccm_correction=True)
    cv2.imwrite('img_ccm.png',image_ccm[...,::-1]**(1/2.2)*255.)
    deltaC, deltaE = icc_minimize.evaluate_result(image_ccm, "linear")
    plt.figure()
    plt.imshow(image_ccm**(1/2.2))
    image_with_gt = icc_minimize.draw_gt_in_image(image_ccm, "linear", deltaE)
    image_with_gt = np.clip(image_with_gt, 0, 1)
    cv2.imwrite('img_ccm_gt.png', image_with_gt[...,::-1]**(1/2.2)*255.)
    

    print('deltaC, deltaE:  ', deltaC.mean(), deltaE.mean())
    # plt.show()
    ccm_cur = icc_minimize.ccm
    cct_cur = icc_minimize.cct
    np.savez(savez_path, cct=cct_cur, ccm=ccm_cur)

def generate_weight(min_weight, max_weight, color):
    ideal_lab = np.float32(np.loadtxt("./data/real_lab_imatest.csv", delimiter=','))  # from imatest
    ideal_xy = smv_colour.XYZ2xyY(smv_colour.Lab2XYZ(torch.from_numpy(ideal_lab))).numpy()[:, 0:2]
    if color == "r":
        c = ideal_xy[14]
    if color == "g":
        c = ideal_xy[13]
    if color == "b":
        c = ideal_xy[12]
    distance = np.sum(np.abs(ideal_xy - c), axis=1)
    weight = max_weight - (max_weight-min_weight) * distance / distance.max()
    return weight

if __name__ == '__main__':
    ccm_space='linear' # srgb or linear
    gt_form='imatest' # imatest or xrite
    savez_path = f"./data/{gt_form}/{ccm_space}/"
    if not os.path.exists(savez_path): os.makedirs(savez_path)

    # image_d65 = cv2.imread(r"./data/mindvision/d65_colorchecker.jpg")[..., ::-1] / 255.
    # savez_path_d65 = savez_path + "D65_light.npz"

    image_A = cv2.imread(r"./data/mindvision/exposure30.jpg")[..., ::-1] / 255.
    savez_path_A = savez_path + "A_light.npz"

    # image_A = cv2.imread(r"./data/tmp/raw_616997531.png")[..., ::-1] / 255.
    # savez_path_A = savez_path + "A_light.npz"

    ccm_weight = np.array([1, 1, 1, 1, 1, 1, 
                           1, 10, 1, 1, 1, 1,
                           10, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1])

    ccm_weight = np.zeros((24))
    # ccm_weight[1] = 1
    ccm_weight[6] = 1
    ccm_weight[11] = 1
    ccm_weight[14] = 1
    ccm_weight[18] = 1



    # gwt = generate_weight(1, 3, 'r')
    # gwt[14] = 5


    generate_calib(image_calib=image_A,
                   ccm_space=ccm_space,
                   image_color_space='linear',
                   gt_form=gt_form,
                   savez_path=savez_path_A,
                   ccm_weight=ccm_weight)
    exit()
