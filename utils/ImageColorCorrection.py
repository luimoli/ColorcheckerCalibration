import sys
sys.path.append("./utils")
import torch
import numpy as np
from misc import gamma, gamma_reverse
import matplotlib.pyplot as plt
import smv_colour
import mcc_detect_color_checker

class ImageColorCorrection:
    def __init__(self, cct_ccm_dict, ccm_cs, method):
        """
        Args:
            cct_ccm_dict:
            method: cc:colorchecker  wp:white paper  grey world:grey world
            white_balance_method:
        """
        # self.__white_balance_method = white_balance_method
        # self.__config = config
        self.__rgb_gain = np.array([1, 1, 1])
        self.__cct = None
        self.__ccm = None
        self.__ccm_cs = ccm_cs
        # self.__image_cs = 'linear'
        self.__method = method
        self.__cct_ccm_dict = cct_ccm_dict

    def setMethod(self, method):
        assert method == 'wp' or method == 'cc', " "
        self.__method = method

    def setWhitePaperGain(self, gain):
        assert self.__method == "wp", "  "
        self.__rgb_gain = self.__rgb_gain * gain

    def ccm_interpolation(self, cct):
        cct_list = sorted(self.__cct_ccm_dict.keys())
        print(cct_list, cct)
        if cct <= cct_list[0]:
            self.__ccm = self.__cct_ccm_dict[cct_list[0]]
        elif cct >= cct_list[-1]:
            self.__ccm = self.__cct_ccm_dict[cct_list[-1]]
        else:
            for i in range(1, len(cct_list)):
                if float(cct) <= cct_list[i]:
                    cct_left = cct_list[i-1]
                    cct_right = cct_list[i]
                    ccm_left = self.__cct_ccm_dict[cct_list[i - 1]]
                    ccm_right = self.__cct_ccm_dict[cct_list[i]]
                    alpha = (1/cct - 1/cct_right) / (1/cct_left - 1/cct_right)
                    print("alpha:", alpha)
                    self.__ccm = alpha * ccm_left + (1-alpha) * ccm_right
                    print(self.__ccm)
                    break

    def compute_cct_from_white_point(self, white_point):
        xyY = smv_colour.XYZ2xyY(smv_colour.RGB2XYZ(torch.from_numpy(np.float32(white_point)), "bt709"))
        cct = smv_colour.xy2CCT(xyY[0:2])
        return float(cct)

    def whitePaperWhiteBalance(self, wb_image):
        height, width = wb_image.shape[0], wb_image.shape[1]
        wp_mean = np.mean(wb_image[int(height*3/8):int(height*5/8), int(width*3/8):int(width*5/8)], axis=(0, 1))
        wb_gain = np.max(wp_mean) / wp_mean
        white_point = 1 / wb_gain
        cct = self.compute_cct_from_white_point(white_point)
        return wb_gain, cct, white_point

    def colorCheckerWhiteBalance(self, wb_image):
        _, charts_rgb, marker_image = mcc_detect_color_checker.detect_colorchecker(wb_image)
        white_block = charts_rgb[18, 0]
        wb_gain = np.max(white_block) / white_block
        white_point = 1 / wb_gain
        cct = self.compute_cct_from_white_point(white_point)
        return wb_gain, cct, white_point

    def multipleLightWhitePaperWhiteBalance(self, wb_image):
        wb_gain = wb_image[..., 1:2] / wb_image
        white_point = 1 / wb_gain
        cct = self.compute_cct_from_white_point(white_point)
        return wb_gain, cct, white_point

    def doWhiteBalance(self, wb_image):
        if self.__method.lower() == "wp":
            rgb_gain, cct, white_point = self.whitePaperWhiteBalance(wb_image)
        elif self.__method.lower() == "cc":
            rgb_gain, cct, white_point = self.colorCheckerWhiteBalance(wb_image)
        elif self.__method.lower() == "multiple_light":
            rgb_gain, cct, white_point = self.multipleLightWhitePaperWhiteBalance(wb_image)
        else:
            assert 0, ""
        self.__rgb_gain = rgb_gain
        self.__cct = cct

    def apply_wb_and_ccm(self, image, image_color_space):
        image_temp = image.copy()
        print(self.__rgb_gain)
        image_temp = image_temp * self.__rgb_gain[None, None]
        image_temp = np.clip(image_temp, 0, 1)
        if image_color_space.lower() == "srgb" and self.__ccm_cs.lower() == "linear":
            image_temp = gamma(image_temp)
        elif image_color_space.lower() == "linear" and self.__ccm_cs.lower() == "srgb":
            image_temp = gamma_reverse(image_temp)

        # apply ccm
        print(self.__ccm.shape)
        # self.__ccm = self.__ccm.numpy()
        if self.__ccm.shape[0] == 4:
            # image_temp = np.einsum('ic, hwc->hwi', self.__ccm[0:3].T, image_temp) + self.__ccm[3][None, None]
            image_temp = np.einsum('ic, hwc->hwi', self.__ccm[0:3].T, image_temp) + self.__ccm[3]
        else:
            image_temp = np.einsum('ic, hwc->hwi', self.__ccm.T, image_temp)
        image_temp = np.clip(image_temp, 0, 1)

        if image_color_space.lower() == "srgb" and self.__ccm_cs.lower() == "linear":
            image_temp = gamma_reverse(image_temp)
        elif image_color_space.lower() == "linear" and self.__ccm_cs.lower() == "srgb":
            image_temp = gamma(image_temp)
        return image_temp

    def correctImage(self, image, image_color_space):
        self.ccm_interpolation(self.__cct)
        corrected_image = self.apply_wb_and_ccm(image, image_color_space)
        return corrected_image
