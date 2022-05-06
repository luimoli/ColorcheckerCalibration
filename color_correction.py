from utils import minimize_ccm
from utils.ImageColorCalibration import ImageColorCorrection
import cv2.cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def get_CCM_interpolation_param(ccm_space, gt_form, calib_name_A, calib_name_B):
    savez_A = np.load(f'./data/{gt_form}/{ccm_space}/{calib_name_A}.npz')
    savez_B = np.load(f'./data/{gt_form}/{ccm_space}/{calib_name_B}.npz')
    cct_A, ccm_A = savez_A["cct"], savez_A["ccm"]
    cct_B, ccm_B = savez_B["cct"], savez_B["ccm"]
    config_predict = {"method": "predict", "ccm_space": ccm_space, "cct1": cct_A, "cct2": cct_B,
                  "ccm1": ccm_A, "ccm2": ccm_B}
    return config_predict

# if ccm_space.lower() == "srgb":
#     cct_A, ccm_A = np.load("./data/xrite/sRGB/A_light.npz")["cct"], np.load("./data/xrite/sRGB/A_light.npz")["ccm"]
#     cct_d65, ccm_d65 = np.load("./data/xrite/sRGB/D65_light.npz")["cct"], np.load("./data/xrite/sRGB/D65_light.npz")["ccm"]
# else:
#     cct_A, ccm_A = np.load("./data/xrite/linearRGB/A_light.npz")["cct"], np.load("./data/xrite/linearRGB/A_light.npz")["ccm"]
#     cct_d65, ccm_d65 = np.load("./data/xrite/linearRGB/D65_light.npz")["cct"], np.load("./data/xrite/linearRGB/D65_light.npz")["ccm"]

# config_predict = {"method": "predict", "ccm_space": ccm_space, "cct1": cct_A, "cct2": cct_d65,
#                   "ccm1": ccm_A, "ccm2": ccm_d65}

if __name__ == '__main__':
    ccm_space = "srgb"   # srgb or linear 
    gt_form = 'imatest'  # imatest or xrite
    calib_name_A, calib_name_B = 'A_light', 'D65_light'
    config_predict = get_CCM_interpolation_param(ccm_space, gt_form, calib_name_A, calib_name_B)

    imgs_path = r'data/mindvision/screen/*.PNG'
    save_root = r'data/mindvision/screen_result_sRGB/'
    if not os.path.exists(save_root): os.makedirs(save_root)

    # for image_path in glob.glob("C:/Users/30880/Desktop/aa/*.PNG")[::2]:
    for image_path in glob.glob(imgs_path)[::2]:
        print(image_path)
        image = cv2.imread(image_path)[..., ::-1] / 255.
        icc_predict = ImageColorCorrection(config_predict, gt_form='imatest', show_log=False)
        icc_predict.update_message_from_colorchecker(image)
        image_ccm = icc_predict.image_correction(image, "linear", white_balance=True, illumination_gain=True,
                                                ccm_correction=True)
        deltaC, deltaE = icc_predict.evaluate_result(image_ccm, "linear")
        image_with_gt = icc_predict.draw_gt_in_image(image_ccm, "linear", deltaE)
        print(deltaC.mean(), deltaE.mean())

        cv2.imwrite(os.path.join(save_root, os.path.basename(image_path)), np.uint8(minimize_ccm.gamma(image_ccm)[...,::-1].copy()*255.))
        cv2.imwrite(os.path.join(save_root, os.path.basename(image_path)[:-4]+'_gt.png'), np.uint8(minimize_ccm.gamma(image_with_gt)[...,::-1].copy()*255.))

        # plt.figure()
        # plt.imshow(minimize_ccm.gamma(image_ccm))
        # plt.show()

