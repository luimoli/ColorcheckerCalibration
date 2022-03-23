import os
from utils.CCM_function2 import ImageColorCorrection
import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt


def generate_calib(image_calib, ccm_space, image_color_space, gt_form, savez_path):
    """This should generate the standalized calibrated CCM.
        'image_calib' is collected under specific illuminant.

    Args:
        image_calib (array): sensor-RGB-IMG
        ccm_space (str): ['srgb','linear']
        image_color_space (str): color-space of the collected image which is used for calibration. ['srgb','linear']
        gt_form (str): decide  ['xrite', 'imatest]
        savez_path (str): save the calibrated CCM and CCT.
    """
    config_minimize = {"method": "minimize", "ccm_space": ccm_space}
    icc_minimize = ImageColorCorrection(config_minimize, gt_form)
    icc_minimize.update_message_from_colorchecker(image_calib)
    image_ccm = icc_minimize.image_correction(image_calib, image_color_space, white_balance=True, illumination_gain=True,
                                                    ccm_correction=True)
    cv2.imwrite('img_ccm.png',image_ccm[...,::-1]**(1/2.2)*255.)
    deltaC, deltaE = icc_minimize.evaluate_result(image_ccm, "linear")

    image_with_gt = icc_minimize.draw_gt_in_image(image_ccm, "linear", deltaE)
    image_with_gt = np.clip(image_with_gt, 0, 1)
    cv2.imwrite('img_ccm_gt.png', image_with_gt[...,::-1]**(1/2.2)*255.)


    print('deltaC, deltaE:  ', deltaC.mean(), deltaE.mean())
    ccm_cur = icc_minimize.ccm
    cct_cur = icc_minimize.cct
    np.savez(savez_path, cct=cct_cur, ccm=ccm_cur)


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
    # plt.figure()
    # plt.imshow(image_A)
    # savez_path_A = savez_path + "A_light.npz"


    generate_calib(image_calib=image_A, ccm_space=ccm_space, image_color_space='linear', gt_form=gt_form, savez_path=savez_path_A)
    # plt.show()
    
    exit()

