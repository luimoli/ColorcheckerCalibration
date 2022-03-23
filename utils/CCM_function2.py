import glob
import torch
import numpy as np
import colour
import cv2.cv2 as cv2

from utils import minimize_ccm
from utils import smv_colour
# from utils import detect_color_checker
from utils import mcc_detect_color_checker
from utils.minimize_ccm import ccm_calculate
from utils.deltaE.deltaC_2000_np import delta_C_CIE2000


ideal_lab_1 = np.float32(np.loadtxt("./data/real_lab_xrite.csv", delimiter=',')) # from x-rite

ideal_lab_2 = np.float32(np.loadtxt("./data/real_lab_imatest.csv", delimiter=','))  # from imatest


class ImageColorCorrection:
    def __init__(self, config, gt_form, show_log=False):
        """_summary_

        Args:
            config (_type_): 
            gt_form (str): ['xrite', 'imatest]
            show_log (bool, optional): . Defaults to False.
        """
        self.config = config
        self.show_log = show_log
        self.rgb_gain = np.array([1, 1, 1])
        self.illumination_gain = 1
        self.cct = None
        self.ccm = np.eye(3, k=1)

        self.ideal_lab = ideal_lab_1 if gt_form == 'xrite' else ideal_lab_2
        self.ideal_linear_rgb = smv_colour.XYZ2RGB(smv_colour.Lab2XYZ(torch.from_numpy(self.ideal_lab)), 'bt709').numpy()
        self.ideal_srgb = minimize_ccm.gamma(self.ideal_linear_rgb)
        self.sorted_centroid = None


    def calculate_colorchecker_value(self, image, sorted_centroid, length):
        sorted_centroid2 = np.int32(sorted_centroid)
        mean_value = np.empty((sorted_centroid.shape[0], 3))
        for i in range(len(sorted_centroid)):
            mean_value[i] = np.mean(image[sorted_centroid2[i, 1] - length:sorted_centroid2[i, 1] + length,
                                    sorted_centroid2[i, 0] - length:sorted_centroid2[i, 0] + length], axis=(0, 1))
        return np.float32(mean_value)

    def compute_rgb_gain_from_colorchecker(self, image, sorted_centroid):
        assert image.max() <= 1, "image range should be in [0, 1]"
        mean_value = self.calculate_colorchecker_value(image, sorted_centroid, 50)
        gain = np.max(mean_value[18:], axis=1)[:, None] / mean_value[18:, ]
        rgb_gain = gain[0:3].mean(axis=0)
        return rgb_gain, mean_value

    def image_white_balance(self, image, rgb_gain):
        assert image.max() <= 1, "image range should be in [0, 1]"
        return image * rgb_gain[None, None]

    def predict_ccm_micheal(self, cct, cct1, cct2, color_matrix1, color_matrix2):
        g = (cct**-1 - cct2**-1) / (cct1**-1 - cct2**-1)
        g = min(1, g)
        g = max(0, g)
        if self.show_log:
            print("cct alpha:", g)
        predicted_ccm = g * color_matrix1 + (1-g) * color_matrix2
        return predicted_ccm

    def draw_groundtruth(self, image, sorted_centroid, groundtruth, length):
        sorted_centroid = np.int32(sorted_centroid)
        image_gt = np.copy(image)
        for i in range(len(sorted_centroid)):
            image_gt[sorted_centroid[i, 1] - length:sorted_centroid[i, 1] + length,
            sorted_centroid[i, 0] - length:sorted_centroid[i, 0] + length] = groundtruth[i]
        return image_gt

    @staticmethod
    def rgb2lab(data):
        assert data.max() <= 1, "image range should be in [0, 1]"
        data = np.float32(data)
        resultxyz = smv_colour.RGB2XYZ(torch.from_numpy(data), 'bt709')
        resultlab = smv_colour.XYZ2Lab(resultxyz).numpy()
        return resultlab

    def compute_cct_from_white_point(self, white_point):
        xyY = smv_colour.XYZ2xyY(smv_colour.RGB2XYZ(torch.from_numpy(np.float32(white_point)), "bt709"))
        cct = colour.xy_to_CCT(xyY[0:2])
        return cct

    def update_message_from_colorchecker(self, image):
        assert image.max() <= 1, "image range should be in [0, 1]"
        self.sorted_centroid, clusters, marker_image = mcc_detect_color_checker.detect_color_checker(image)
        self.rgb_gain, cc_mean_value = self.compute_rgb_gain_from_colorchecker(image, self.sorted_centroid)
        self.cct = self.compute_cct_from_white_point(1/self.rgb_gain)
        cc_wb_mean_value = cc_mean_value * self.rgb_gain[None]
        self.illumination_gain = (self.ideal_linear_rgb[18:21] / cc_wb_mean_value[18:21]).mean()
        cc_wb_ill_mean_value = self.illumination_gain * cc_wb_mean_value

        if self.show_log:
            print("cct:", self.cct)
            print("rgb_gain:", self.rgb_gain)
            print("illumination_gain:", self.illumination_gain)

        if self.config["method"] == "minimize":
            if self.config["ccm_space"].lower() == "srgb":
                cc_wb_ill_mean_value = minimize_ccm.gamma(cc_wb_ill_mean_value)
            self.ccm = ccm_calculate(cc_wb_ill_mean_value, self.ideal_lab, self.config["ccm_space"])
            
        if self.config["method"] == "predict":
            self.ccm = self.predict_ccm_micheal(self.cct, self.config["cct1"], self.config["cct2"],
                                                self.config["ccm1"], self.config["ccm2"])

    def image_correction(self, image, image_color_space, white_balance=True,
                         illumination_gain=True, ccm_correction=True):
        print(self.rgb_gain)
        if white_balance:
            image = image * self.rgb_gain[None, None]
            image = np.clip(image, 0, 1)
        if illumination_gain:
            image = image * self.illumination_gain
            image = np.clip(image, 0, 1)

        if ccm_correction:
            if self.show_log:
                print(image_color_space.lower(), self.config["ccm_space"].lower())

            if image_color_space.lower() == self.config["ccm_space"].lower():
                image = np.einsum('ic, hwc->hwi', self.ccm, image)

            elif image_color_space.lower() == "linear" and self.config["ccm_space"].lower() == "srgb":
                image = minimize_ccm.gamma(image, "sRGB")
                image = np.einsum('ic, hwc->hwi', self.ccm, image)
                image = minimize_ccm.gamma_reverse(image, "sRGB")

            elif image_color_space.lower() == "srgb" and self.config["ccm_space"].lower() == "linear":
                image = minimize_ccm.gamma_reverse(image, "sRGB")
                image = np.einsum('ic, hwc->hwi', self.ccm, image)
                image = minimize_ccm.gamma(image, "sRGB")
                
            image = np.clip(image, 0, 1)

        return image

    def evaluate_result(self, image, image_color_space):
        if self.sorted_centroid is None:
            self.sorted_centroid, clusters, marker_image = mcc_detect_color_checker.detect_color_checker(image)
        result_cc_mean = self.calculate_colorchecker_value(image, self.sorted_centroid, 50)
        result_cc_mean = np.clip(result_cc_mean, 0, 1)
        if image_color_space == "srgb":
            result_cc_mean_lab = self.rgb2lab(minimize_ccm.gamma_reverse(result_cc_mean))
        else:
            result_cc_mean_lab = self.rgb2lab(result_cc_mean)
        deltaC = delta_C_CIE2000(result_cc_mean_lab, self.ideal_lab)
        deltaE = colour.delta_E(result_cc_mean_lab, self.ideal_lab, 'CIE 2000')
        return deltaC, deltaE

    def draw_gt_in_image(self, image, image_color_space, deltaE):
        if self.sorted_centroid is None:
            self.sorted_centroid, clusters, marker_image = mcc_detect_color_checker.detect_color_checker(image)
        image_gt = image.copy()
        length = 50
        self.sorted_centroid = np.int32(self.sorted_centroid)
        for i in range(len(self.sorted_centroid)):
            if image_color_space.lower() == "linear":
                image_gt[self.sorted_centroid[i, 1] - length:self.sorted_centroid[i, 1] + length,
                self.sorted_centroid[i, 0] - length:self.sorted_centroid[i, 0] + length] = self.ideal_linear_rgb[i]
            else:
                image_gt[self.sorted_centroid[i, 1] - length:self.sorted_centroid[i, 1] + length,
                self.sorted_centroid[i, 0] - length:self.sorted_centroid[i, 0] + length] = self.ideal_srgb[i]
            cv2.putText(image_gt, str(round(deltaE[i], 1)), np.int32(self.sorted_centroid[i]),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)
        return image_gt


if __name__ == '__main__':
    image_d65 = cv2.imread(r"E:/data/mindvision/d65/d65_colorchecker.jpg")[..., ::-1] / 255.
    image_A = cv2.imread(r"E:/data/mindvision/A_light/exposure30.jpg")[..., ::-1] / 255.
    image_tl84 = cv2.imread(r"E:/data/mindvision/tl84/tl84_colorchecker.jpg")[..., ::-1] / 255.
    image_cwf = cv2.imread(r"E:/data/mindvision/CWF/CWF_colorchecker.jpg")[..., ::-1] / 255.

    data = np.load("d65_A_cct_ccm_srgb2.npz")
    cct_A = data["cct_A"]
    cct_d65 = data["cct_d65"]
    ccm_A = data["ccm_A"]
    ccm_d65 = data["ccm_d65"]

    data = np.load("d65_A_cct_ccm_linear.npz")
    cct_A_linear = data["cct_A"]
    cct_d65_linear = data["cct_d65"]
    ccm_A_linear = data["ccm_A"]
    ccm_d65_linear = data["ccm_d65"]
    #
    # print(cct_A, cct_d65)
    # exit()

    config_minimize_linear = {"method": "minimize", "ccm_space": "linear"}
    config_minimize_srgb = {"method": "minimize", "ccm_space": "srgb"}
    icc_minimize_srgb = ImageColorCorrection(config_minimize_srgb)
    icc_minimize_linear = ImageColorCorrection(config_minimize_linear)
    '''
    icc_minimize_srgb.update_message_from_colorchecker(image_A)
    image_ccm = icc_minimize_srgb.image_correction(image_A, "linear", white_balance=True, illumination_gain=True,
                                                     ccm_correction=True)
    deltaC, deltaE = icc_minimize_srgb.evaluate_result(image_ccm, "linear")
    print(deltaC.mean(), deltaE.mean())
    ccm_A = icc_minimize_srgb.ccm
    cct_A = icc_minimize_srgb.cct

    icc_minimize_srgb.update_message_from_colorchecker(image_d65)
    image_ccm = icc_minimize_srgb.image_correction(image_d65, "linear", white_balance=True, illumination_gain=True,
                                                     ccm_correction=True)
    deltaC, deltaE = icc_minimize_srgb.evaluate_result(image_ccm, "linear")
    print(deltaC.mean(), deltaE.mean())

    ccm_d65 = icc_minimize_srgb.ccm
    cct_d65 = icc_minimize_srgb.cct
    np.savez("d65_A_cct_ccm_srgb2.npz", cct_A=cct_A, cct_d65=cct_d65, ccm_A=ccm_A, ccm_d65=ccm_d65)
    exit()
    '''

    config_predict_srgb = {"method": "predict", "ccm_space": "srgb", "cct1": cct_A, "cct2": cct_d65,
                           "ccm1": ccm_A, "ccm2": ccm_d65}

    config_predict_linear = {"method": "predict", "ccm_space": "linear", "cct1": cct_A_linear, "cct2": cct_d65_linear,
                             "ccm1": ccm_A_linear, "ccm2": ccm_d65_linear}

    # icc = ImageColorCorrection(config_minimize_srgb)
    # for image_path in glob.glob("C:/Users/30880/Desktop/basler/*"):
    #     image = cv2.imread(image_path)[..., ::-1] / 255.
    #     icc = ImageColorCorrection(config_minimize_srgb)
    #     deltaC, deltaE = icc.evaluate_result(image, "srgb")
    #     print(deltaC.mean(), deltaC.max(), end=" ")
    #     print(deltaE.mean(), deltaE.max())

    # for image_path in glob.glob("C:/Users/30880/Desktop/hk/*"):
    #     image = cv2.imread(image_path)[..., ::-1] / 255.
    #     icc = ImageColorCorrection(config_minimize_srgb)
    #     deltaC, deltaE = icc.evaluate_result(image, "srgb")
    #     print(deltaC.mean(), deltaC.max(), end=" ")
    #     print(deltaE.mean(), deltaE.max())
    # exit()

    for image_path in glob.glob("C:/Users/30880/Desktop/aa/*.PNG")[::2]:
        print(image_path)
        image_cwf = cv2.imread(image_path)[..., ::-1] / 255.
        icc_predict_srgb = ImageColorCorrection(config_predict_srgb, False)
        icc_predict_linear = ImageColorCorrection(config_predict_linear, False)
        icc_predict_linear.update_message_from_colorchecker(image_cwf)
        image_ccm = icc_predict_linear.image_correction(image_cwf, "linear", white_balance=True, illumination_gain=True, ccm_correction=True)
        deltaC, deltaE = icc_predict_linear.evaluate_result(image_ccm, "linear")
        image_gt = icc_predict_linear.draw_gt_in_image(image_ccm, "linear", deltaE)
        cv2.imwrite(image_path.replace(".PNG", "linear_ccm_with_gt.jpeg"), minimize_ccm.gamma(image_gt)[..., ::-1] * 255)

        # plt.figure()
        # plt.imshow(image_gt)
        # plt.show()


        print("linear", deltaC.mean(), deltaC.max(), end=" ")
        print(deltaE.mean(), deltaE.max())
        icc_predict_srgb.update_message_from_colorchecker(image_cwf)
        image_ccm = icc_predict_srgb.image_correction(image_cwf, "linear", white_balance=True, illumination_gain=True, ccm_correction=True)
        deltaC, deltaE = icc_predict_srgb.evaluate_result(image_ccm, "linear")
        image_gt = icc_predict_linear.draw_gt_in_image(image_ccm, "linear", deltaE)
        print("srgb", deltaC.mean(), deltaC.max(), end=" ")
        print(deltaE.mean(), deltaE.max())

        cv2.imwrite(image_path.replace(".PNG", "srgb_ccm_with_gt.jpeg"), minimize_ccm.gamma(image_gt)[..., ::-1] * 255)

