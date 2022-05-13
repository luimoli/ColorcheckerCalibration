import sys
import cv2
sys.path.append("./utils")
import numpy as np
from misc import gamma, gamma_reverse
import matplotlib.pyplot as plt
import smv_colour
import mcc_detect_color_checker


class ImageColorCorrection:
    def __init__(self, cct_ccm_dict, ccm_cs):
        """
        Args:
            cct_ccm_dict:
            method: cc:colorchecker  wp:white paper  grey world:grey world
            white_balance_method:
        """
        self.rgb_gain = np.array([1, 1, 1])
        self.cct = None
        self.ccm = None
        self.ccm_cs = ccm_cs
        self.method = None
        self.cct_ccm_dict = cct_ccm_dict

    def setMethod(self, method):
        # assert method == 'wp' or method == 'cc', " "
        self.method = method

    def setWhitePaperGain(self, gain):
        assert self.method == "wp", "  "
        self.rgb_gain = self.rgb_gain * gain

    def ccm_interpolation(self, cct):
        cct_list = sorted(self.cct_ccm_dict.keys())
        print(cct.shape)
        if cct <= cct_list[0]:
            self.ccm = self.cct_ccm_dict[cct_list[0]]
        elif cct >= cct_list[-1]:
            self.ccm = self.cct_ccm_dict[cct_list[-1]]
        else:
            for i in range(1, len(cct_list)):
                if float(cct) <= cct_list[i]:
                    cct_left = cct_list[i-1]
                    cct_right = cct_list[i]
                    ccm_left = self.cct_ccm_dict[cct_list[i - 1]]
                    ccm_right = self.cct_ccm_dict[cct_list[i]]
                    alpha = (1/cct - 1/cct_right) / (1/cct_left - 1/cct_right)
                    print("alpha:", alpha)
                    self.ccm = alpha * ccm_left + (1-alpha) * ccm_right
                    print(self.ccm)
                    break

    def compute_cct_from_white_point(self, white_point):
        xyY = smv_colour.XYZ2xyY(smv_colour.RGB2XYZ(np.float32(white_point), "bt709"))
        cct = smv_colour.xy2CCT(xyY[..., 0:2])
        return np.float32(cct)

    def whitePaperWhiteBalance(self, wb_image, show_wb_area=False):
        height, width = wb_image.shape[0], wb_image.shape[1]
        wp_mean = np.mean(wb_image[int(height*2/8):int(height*4/8), int(width*3/8):int(width*5/8)], axis=(0, 1))
        wb_gain = np.max(wp_mean) / wp_mean
        white_point = 1 / wb_gain
        if show_wb_area:
            wb_image = cv2.rectangle(wb_image, (int(width*3/8), int(height*2/8)), (int(width*5/8), int(height*4/8)), (0, 0, 0), 2)
            plt.figure()
            plt.imshow(wb_image)
            plt.show()
        cct = self.compute_cct_from_white_point(white_point)
        return wb_gain, cct, white_point

    def colorCheckerWhiteBalance(self, wb_image):
        _, _, charts_rgb, marker_image = mcc_detect_color_checker.detect_colorchecker(wb_image)
        white_block = charts_rgb[18]
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
        if self.method.lower() == "wp":
            rgb_gain, cct, white_point = self.whitePaperWhiteBalance(wb_image)
        elif self.method.lower() == "cc":
            rgb_gain, cct, white_point = self.colorCheckerWhiteBalance(wb_image)
        elif self.method.lower() == "multiple_light":
            rgb_gain, cct, white_point = self.multipleLightWhitePaperWhiteBalance(wb_image)
        self.rgb_gain = rgb_gain
        self.cct = cct
        print(self.rgb_gain, self.cct )

    def apply_wb_and_ccm(self, image, image_color_space):
        image_temp = image.copy()
        print(self.rgb_gain)
        image_temp = image_temp * self.rgb_gain[None, None]

        image_temp = np.clip(image_temp, 0, 1)
        if image_color_space.lower() == "srgb" and self.ccm_cs.lower() == "linear":
            image_temp = gamma(image_temp)
        elif image_color_space.lower() == "linear" and self.ccm_cs.lower() == "srgb":
            image_temp = gamma_reverse(image_temp)

        # apply ccm
        print(self.ccm.shape)
        self.ccm = self.ccm.T
        # self.__ccm = self.__ccm.numpy()
        if self.ccm.shape[0] == 4:
            # image_temp = np.einsum('ic, hwc->hwi', self.__ccm[0:3].T, image_temp) + self.__ccm[3][None, None]
            image_temp = np.einsum('ic, hwc->hwi', self.ccm[0:3].T, image_temp) + self.ccm[3][None, None]
        else:
            image_temp = np.einsum('ic, hwc->hwi', self.ccm.T, image_temp)
        image_temp = np.clip(image_temp, 0, 1)

        if image_color_space.lower() == "srgb" and self.ccm_cs.lower() == "linear":
            image_temp = gamma_reverse(image_temp)
        elif image_color_space.lower() == "linear" and self.ccm_cs.lower() == "srgb":
            image_temp = gamma(image_temp)
        return image_temp

    def correctImage(self, image, image_color_space):
        self.ccm_interpolation(self.cct)
        corrected_image = self.apply_wb_and_ccm(image, image_color_space)
        return corrected_image
