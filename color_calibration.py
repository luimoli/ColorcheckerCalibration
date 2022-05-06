import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2.cv2 as cv2

from utils.ImageColorCalibration import ImageColorCorrection
from utils import smv_colour



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

    deltaC, deltaE00, deltaE76 = icc_minimize.evaluate_result(image_ccm, "linear")
    image_with_gt = icc_minimize.draw_gt_in_image(image_ccm, "linear", deltaC)
    image_with_gt = np.clip(image_with_gt, 0, 1)
    cv2.imwrite('img_ccm_gt.png', image_with_gt[...,::-1]**(1/2.2)*255.)

    print('deltaC00, deltaE00: ', deltaC.mean(), deltaE00.mean())
    # print('deltaC[18:], deltaE00[18:], ', deltaC[18:].mean(), deltaE00[18:].mean())

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

    # image = cv2.imread(r"./data/mindvision/d65_colorchecker.jpg")[..., ::-1] / 255.
    # savez_path = savez_path + "D65_light.npz"
    # image = cv2.imread(r"./data/mindvision/exposure30.jpg")[..., ::-1] / 255.
    # savez_path = savez_path + "A_light.npz"

    image = (cv2.imread(r"./data/mindvision/mv_2300.PNG")[..., ::-1] - 0) / 255.
    savez_path = savez_path + "A_light.npz"
    # image = cv2.imread(r"./data/mindvision/mv_8000.PNG")[..., ::-1] / 255.
    # savez_path = savez_path + "D65_light.npz"

    # image = cv2.imread(r"./data/mindvision/mvmv_2300.png", -1) / 65535.
    # savez_path = savez_path + "D65_light.npz"

    # image = cv2.imread(r"./data/tmp/raw_616997531.png")[..., ::-1] / 255.
    # savez_path = savez_path + "A_light.npz"

    

    ccm_weight = np.array([1, 1, 1, 1, 1, 1, 
                           1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1])
    

    # ccm_weight[[i for i in range(1, 25, 2)]] = 0



    generate_calib(image_calib=image,
                   ccm_space=ccm_space,
                   image_color_space='linear',
                   gt_form=gt_form,
                   savez_path=savez_path,
                   ccm_weight=ccm_weight)
    

    # image_ = cv2.imread(r"./data/tmp/image_D65.bmp")[..., ::-1] / 255.
    # config_minimize = {"method": "minimize", "ccm_space": ccm_space, "gt_form": gt_form, "ccm_weight": ccm_weight}
    # icc_minimize = ImageColorCorrection(config_minimize)
    # deltaC, deltaE00, deltaE76 = icc_minimize.evaluate_result(image_, 'srgb')
    # image_with_gt = icc_minimize.draw_gt_in_image(image_, "linear", deltaC)
    # image_with_gt = np.clip(image_with_gt, 0, 1)
    # cv2.imwrite(r'./data/tmp/image_D65_gt.bmp', image_with_gt[...,::-1]*255.)
    # print('deltaC, deltaE00, deltaE76:  ', deltaC.mean(), deltaE00.mean(), deltaE76.mean())

    exit()
