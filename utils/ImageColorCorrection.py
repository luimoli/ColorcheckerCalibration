import torch
import numpy as np
import colour
import cv2.cv2 as cv2
import scipy.optimize as optimize
import matplotlib.pyplot as plt

from utils.deltaE.deltaC_2000_np import delta_C_CIE2000
from utils.deltaE.deltaE_2000_np import delta_E_CIE2000
from utils import smv_colour
# from utils import mcc_detect_color_checker
from utils.mcc_detect_color_checker import detect_colorchecker_value, detect_colorchecker
from utils.misc import gamma, gamma_reverse, rgb2lab


class ImageColorCorrection:
    def __init__(self, config, material='cc' ):
        """_summary_

        Args:
            config (_type_): 
            gt_form (str): ['xrite', 'imatest]
            show_log (bool, optional): . Defaults to False.
        """
        self.config = config
        print(self.config)

        self.rgb_gain = np.array([1, 1, 1])
        self.cct = None
        
        # self.ideal_lab = None
        # self.ideal_linear_rgb = None
        # self.setColorChecker_Lab(mode=2)

        self.__colorspace = 'linear'
        self.__material = material

        self.sorted_centroid = None
        self.cc_mean_value = None

    def setMaterial(self, value):
        assert value == 'wp' or value == 'cc'
        self.__material == value

    def predict_ccm(cct_ccm_dict, cct):
        cct_list = sorted(cct_ccm_dict.keys())
        print(cct_list, cct)
        if cct <= cct_list[0]:
            return cct_ccm_dict[cct_list[0]]
        elif cct >= cct_list[-1]:
            return cct_ccm_dict[cct_list[-1]]
        for i in range(1, len(cct_list)):
            if float(cct) <= cct_list[i]:
                cct_left = cct_list[i-1]
                cct_right = cct_list[i]
                ccm_left = cct_ccm_dict[cct_list[i - 1]]
                ccm_right = cct_ccm_dict[cct_list[i]]
                alpha = (1/cct - 1/cct_right) / (1/cct_left - 1/cct_right)
                ccm = alpha * ccm_left + (1-alpha) * ccm_right
                break
        return ccm

    def compute_cct_from_white_point(self, white_point):
        xyY = smv_colour.XYZ2xyY(smv_colour.RGB2XYZ(torch.from_numpy(np.float32(white_point)), "bt709"))
        cct = smv_colour.xy2CCT(xyY[0:2])
        return cct
    
    def compute_rgb_gain_from_colorchecker(self, mean_value):
        gain = np.max(mean_value[18:], axis=1)[:, None] / mean_value[18:, ]
        rgb_gain = gain[0:3].mean(axis=0)
        return rgb_gain

    def update_message_from_colorchecker(self, image):
        if self.__material == 'cc':
            self.cc_mean_value = detect_colorchecker_value(image)
            self.rgb_gain = self.compute_rgb_gain_from_colorchecker(self.cc_mean_value)
            self.cct = self.compute_cct_from_white_point( 1 / self.rgb_gain)
        elif self.__material == 'wp':
            self.cct = None
            #TODO
            
        # if self.config["method"] == "predict":
        #     self.ccm = self.predict_ccm_micheal(self.cct, self.config["cct1"], self.config["cct2"],
        #                                         self.config["ccm1"], self.config["ccm2"])
        self.ccm = self.predict_ccm(cct_ccm_dict, self.cct)