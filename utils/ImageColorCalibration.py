import torch
import numpy as np
import colour
import cv2.cv2 as cv2
import scipy.optimize as optimize
import matplotlib.pyplot as plt

from utils import smv_colour
from utils.deltaE.deltaC_2000_np import delta_C_CIE2000
from utils.deltaE.deltaE_2000_np import delta_E_CIE2000
from utils.mcc_detect_color_checker import detect_colorchecker_value, detect_colorchecker
from utils.misc import gamma, gamma_reverse, rgb2lab, lab2rgb
from utils.evaluate_result import evaluate

class ImageColorCalibration:
    def __init__(self, src_for_ccm, colorchecker_gt_mode):
        """_summary_

        Args:
            src_for_ccm (arr): src rgb-colorchecker-data(eg.with shape [24,3])
            colorchecker_gt_mode (int): 1: xrite, 2: imatest, 3: 3nh
        """
        # self.config = config
        # print(self.config)
        # self.show_log = show_log

        self.rgb_gain = np.array([1, 1, 1])
        self.illumination_gain = 1
        self.cct = None

        self.ideal_lab = None
        self.ideal_linear_rgb = None
        self.setColorChecker_Lab(mode=colorchecker_gt_mode)

        self.__colorspace = 'linear'
        self.__ccm = np.eye(3, k=1)
        self.__ccm_method = 'minimize'
        self.__ccm_weight = np.ones((self.ideal_lab.shape[0]))
        self.__ccm_masks = src_for_ccm < (240 / 255)
        self.__ccm_type = '3x3'
        self.__ccm_metric = 'CIE2000'
        self.__ccm_rowsum1 = True
        self.__ccm_constrain = 0

        self.sorted_centroid = None
        self.cc_mean_value = src_for_ccm

        # self.ccm_dict_path = ccm_dict_path

    def setColorChecker_Lab(self, mode):
        """set groundtruth of colorchecker.
        Args:
            mode (int): 1: xrite, 2: imatest, 3: 3nh
        """
        ideal_lab_1 = np.float32(np.loadtxt("./data/real_lab_xrite.csv", delimiter=',')) # from x-rite
        ideal_lab_2 = np.float32(np.loadtxt("./data/real_lab_imatest.csv", delimiter=','))  # from imatest
        ideal_lab_3 = np.float32(np.loadtxt("./data/real_lab_d50_3ns.csv", delimiter=','))  # from 3nh
        ideal_lab_dic = {1:ideal_lab_1, 2: ideal_lab_2, 3: ideal_lab_3}
        self.ideal_lab = ideal_lab_dic[mode]
        self.ideal_linear_rgb = lab2rgb(self.ideal_lab)

    def setColorSpace(self, space):
        """set the colorspace when calculating CCM.
        Args:
            space (str): ['srgb', 'linear']
        """
        assert space == 'srgb' or space =='linear'
        self.__colorspace = space

    def setCCM_WEIGHT(self, arr):
        """_summary_
        Args:
            arr (array): the weight for colorcheckers' patches of CCM.
        """
        assert arr.shape[0] == self.ideal_lab.shape[0] and arr.ndim == 1
        self.__ccm_weight = arr

    def getCCM_WEIGHT(self):
        return self.__ccm_weight
    
    def setCCM_MASKS(self, arr):
        assert arr.shape[0] == self.ideal_lab.shape[0] and arr.ndim == 1
        self.__ccm_masks = arr
    
    def getCCM_MASKS(self):
        return self.__ccm_masks


    def setCCM_TYPE(self, type):
        """set the shape of CCM.
        Args:
            type (str): ['3x3', '3x4']
        """
        assert type == '3x3' or type == '3x4'
        self.__ccm_type = type

    def setCCM_METRIC(self, metric):
        """the metric of CCM optimization.
        Args:
            metric (str): ['CIE2000', 'CIE1976]
        """
        assert metric == 'CIE2000' or metric == 'CIE1976'
        self.__ccm_metric = metric

    def setCCM_METHOD(self, method):
        """the method of CCM calculation.
        Args:
            method (str): ['minimize', 'polynominal']
        """
        assert method == 'minimize' or method == 'polynominal'
        self.__ccm_method = method


    def setCCM_RowSum1(self, boolvalue):
        """whether to mantain white balance constrain: the sum of CCM's row is 1.
        Args:
            boolvalue (bool): True or False
        """
        self.__ccm_rowsum1 = boolvalue

    def setCCM_Constrain(self, value):
        """set diagonal value constrain.
        Args:
            value (float): constrain the diagonal of CCM to be less than a value when calculating CCM.
        """
        assert value > 0
        self.__ccm_constrain = value

    def getCCM(self):
        return self.__ccm


    def compute_rgb_gain_from_colorchecker(self, mean_value):
        # assert image.max() <= 1, "image range should be in [0, 1]"
        gain = np.max(mean_value[18:], axis=1)[:, None] / mean_value[18:, ]
        rgb_gain = gain[0:3].mean(axis=0)
        return rgb_gain


    def compute_cct_from_white_point(self, white_point):
        xyz = smv_colour.RGB2XYZ(torch.from_numpy(np.float32(white_point)), "bt709")
        xyY = smv_colour.XYZ2xyY(xyz)
        cct = smv_colour.xy2CCT(xyY[0:2])
        return cct


    def run(self):
        self.rgb_gain = self.compute_rgb_gain_from_colorchecker(self.cc_mean_value)
        self.cct = self.compute_cct_from_white_point( 1 / self.rgb_gain)

        cc_wb_mean_value = self.cc_mean_value * self.rgb_gain[None]
        self.illumination_gain = (self.ideal_linear_rgb[18:21] / cc_wb_mean_value[18:21]).mean()
        cc_wb_ill_mean_value = self.illumination_gain * cc_wb_mean_value
        
        self.__ccm_masks = cc_wb_ill_mean_value < 1
        cc_wb_ill_mean_value = cc_wb_ill_mean_value * self.__ccm_masks[..., None]
        

        if self.__ccm_method == "minimize":
            if self.__colorspace.lower() == "srgb":
                cc_wb_ill_mean_value = gamma(cc_wb_ill_mean_value)
            if self.__ccm_type == '3x4':
                cc_wb_ill_mean_value = np.concatenate((cc_wb_ill_mean_value.copy(), np.ones((cc_wb_ill_mean_value.shape[0], 1))), axis=-1)
                # print(cc_wb_ill_mean_value.shape)
            self.__ccm = self.ccm_calculate(cc_wb_ill_mean_value)

        if self.__ccm_method == 'polynominal':
            cc_wb_ill_mean_value2 = np.empty((24, 9))
            cc_wb_ill_mean_value2[:, 0:3] = cc_wb_ill_mean_value
            cc_wb_ill_mean_value2[:, 3] = cc_wb_ill_mean_value[:, 0] * cc_wb_ill_mean_value[:, 0]
            cc_wb_ill_mean_value2[:, 4] = cc_wb_ill_mean_value[:, 1] * cc_wb_ill_mean_value[:, 1]
            cc_wb_ill_mean_value2[:, 5] = cc_wb_ill_mean_value[:, 2] * cc_wb_ill_mean_value[:, 2]
            cc_wb_ill_mean_value2[:, 6] = cc_wb_ill_mean_value[:, 0] * cc_wb_ill_mean_value[:, 1]
            cc_wb_ill_mean_value2[:, 7] = cc_wb_ill_mean_value[:, 0] * cc_wb_ill_mean_value[:, 2]
            cc_wb_ill_mean_value2[:, 8] = cc_wb_ill_mean_value[:, 1] * cc_wb_ill_mean_value[:, 2]
            self.__ccm = self.ccm_calculate(cc_wb_ill_mean_value2)


    def apply_ccm(self, img, ccm):
        assert ccm.shape[0] == 3
        if img.ndim == 3:
            img_ccm = np.einsum('hwi,ji->hwj', img, ccm)
        elif img.ndim == 2:
            img_ccm = np.einsum('hi,ji->hj', img, ccm)
        else:
            raise ValueError(img.shape)
        return img_ccm

    def ccm_calculate(self, rgb_data):
        """[calculate the color correction matrix]
        Args:
            rgb_data ([N*3]): [the RGB data of color_checker]
        Returns:
            [array]: [CCM with shape: 3*3 or 3*4]
        """
        if self.__ccm_method.lower() == 'minimize':
            if self.__ccm_type == '3x3':
                if self.__ccm_rowsum1:
                    x2ccm=lambda x : np.array([[1-x[0]-x[1],x[0],x[1]],
                                            [x[2],1-x[2]-x[3],x[3]],
                                            [x[4],x[5],1-x[4]-x[5]]])
                    x0 = np.zeros((6))
                    
                else:
                    x2ccm=lambda x : np.array([[x[0], x[1],x[2]],
                                            [x[3], x[4], x[5]],
                                            [x[6],x[7],x[8]]])
                    # x0 = np.zeros((9))
                    x0 = np.array([[self.ideal_linear_rgb[..., 0].mean() / rgb_data[..., 0].mean(), 0, 0],
                                   [0, self.ideal_linear_rgb[..., 1].mean() / rgb_data[..., 1].mean(), 0],
                                   [0, 0, self.ideal_linear_rgb[..., 2].mean() / rgb_data[..., 2].mean()]])

            elif self.__ccm_type == '3x4':
                if self.__ccm_rowsum1:
                    x2ccm=lambda x : np.array([[1-x[0]-x[1],x[0],x[1], x[6]],
                                               [x[2],1-x[2]-x[3],x[3], x[7]],
                                               [x[4],x[5],1-x[4]-x[5], x[8]]])
                    x0 = np.zeros((9))
                else:
                    x2ccm=lambda x : np.array([[x[0], x[1], x[2], x[3]],
                                               [x[4], x[5], x[6], x[7]],
                                               [x[8], x[9], x[10], x[11]]])
                    x0 = np.array([[self.ideal_linear_rgb[..., 0].mean() / rgb_data[..., 0].mean(), 0, 0, 0],
                                   [0, self.ideal_linear_rgb[..., 1].mean() / rgb_data[..., 1].mean(), 0, 0],
                                   [0, 0, self.ideal_linear_rgb[..., 2].mean() / rgb_data[..., 2].mean(), 0]])

        elif self.__ccm_method.lower() == 'polynominal':
            x2ccm=lambda x : np.array([[x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]],
                                [x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17]],
                                [x[18], x[19], x[20], x[21], x[22], x[23], x[24], x[25], x[26]]])
            x0 = np.zeros((27))

        if self.__colorspace.lower() == "linear":
            f_lab=lambda x : rgb2lab(self.apply_ccm(rgb_data, x2ccm(x)))
        elif self.__colorspace.lower() == "srgb":
            f_lab = lambda x: rgb2lab(gamma_reverse(self.apply_ccm(rgb_data, x2ccm(x)), colorspace='sRGB'))

        if self.__ccm_metric == 'CIE1976':
            f_error=lambda x : f_lab(x)- self.ideal_lab
            f_DeltaE=lambda x : (np.sqrt((f_error(x)**2).sum(axis=1)) * self.__ccm_weight).mean()
        elif self.__ccm_metric == 'CIE2000':
            f_DeltaE=lambda x : ((delta_E_CIE2000(f_lab(x), self.ideal_lab) * self.__ccm_weight)**2).mean()

        func=lambda x : print('deltaE_00 = ',f_DeltaE(x))

        if self.__ccm_constrain:
            if self.__ccm_type == '3x3':
                if self.__ccm_rowsum1:
                    cons = ({'type': 'ineq', 'fun': lambda x: x[0] + x[1] -1 + self.__ccm_constrain},\
                            {'type': 'ineq', 'fun': lambda x: x[2] + x[3] -1 + self.__ccm_constrain},\
                            {'type': 'ineq', 'fun': lambda x: x[4] + x[5] -1 + self.__ccm_constrain})
                else:
                    cons = ({'type': 'ineq', 'fun': lambda x: -x[0] + self.__ccm_constrain},\
                            {'type': 'ineq', 'fun': lambda x: -x[4] + self.__ccm_constrain},\
                            {'type': 'ineq', 'fun': lambda x: -x[8] + self.__ccm_constrain})

                result=optimize.minimize(f_DeltaE, x0, method='SLSQP', constraints=cons)

            elif self.__ccm_type == '3x4':
                raise ValueError('currently not supported constrain value in 3*4 CCM.')

        else:
            # x0 = np.array([1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0])
            # x0 = np.array([1, 0, 0, 0.5, 0, 1, 0, 0.5, 0, 0, 1, 0.5])
            x0 = np.zeros(12)

            result=optimize.minimize(f_DeltaE, x0, callback=func, method='Powell')

        print('minimize average deltaE00: ', result.fun)
        # print('--------------------')
        # print('ccm:\n',x2ccm(result.x))
        # print('--------------------')
        return x2ccm(result.x)



    def infer(self,
              img,
              image_color_space,
              white_balance=True,
              illumination_gain=True,
              ccm_correction=True):
        """infer the img using the calculated CCM.
            The output img's colorspace keeps with parameter 'image_color_space'.  

        Args:
            img (arr): typically with shape[H,W,3].
            image_color_space (str): ['srgb', 'linear']
            white_balance (bool, optional): . Defaults to True.
            illumination_gain (bool, optional): . Defaults to True.
            ccm_correction (bool, optional): . Defaults to True.

        Returns:
            arr: calibrated image.
        """
        # print('rgb_gain:  ',self.rgb_gain)
        image = img.copy()
        if white_balance:
            assert image.max() <= 1
            image = image * self.rgb_gain[None, None]
            image = np.clip(image, 0, 1)

        if illumination_gain:
            image = image * self.illumination_gain
            image = np.clip(image, 0, 1)

        if ccm_correction:
            if self.__ccm_type == '3x4':
                image = np.concatenate((image.copy(), np.ones((image.shape[0], image.shape[1], 1))), axis=-1)
                # print(image.shape)

            if image_color_space.lower() == self.__colorspace.lower():
                # image = np.einsum('ic, hwc->hwi', self.__ccm, image)
                image = self.apply_ccm(image, self.__ccm)

            elif image_color_space.lower() == "linear" and self.__colorspace.lower() == "srgb":
                image = gamma(image, "sRGB")
                # image = np.einsum('ic, hwc->hwi', self.__ccm, image)
                image = self.apply_ccm(image, self.__ccm)
                image = gamma_reverse(image, "sRGB")

            elif image_color_space.lower() == "srgb" and self.__colorspace.lower() == "linear":
                image = gamma_reverse(image, "sRGB")
                # image = np.einsum('ic, hwc->hwi', self.__ccm, image)
                image = self.apply_ccm(image, self.__ccm)
                image = gamma(image, "sRGB")

            image = np.clip(image, 0, 1)

        return image



    # def evaluate_result(self, image, image_color_space):
    #     # if self.sorted_centroid is None:
    #     #     self.sorted_centroid, clusters, marker_image = detect_colorchecker(image)
    #     # result_cc_mean = self.calculate_colorchecker_value(image, self.sorted_centroid, 50)

    #     result_cc_mean = detect_colorchecker_value(image)

    #     result_cc_mean = np.clip(result_cc_mean, 0, 1)
    #     if image_color_space == "srgb":
    #         result_cc_mean_lab = rgb2lab(gamma_reverse(result_cc_mean))
    #     else:
    #         result_cc_mean_lab = rgb2lab(result_cc_mean)
    #     deltaC = delta_C_CIE2000(result_cc_mean_lab, self.ideal_lab)
    #     deltaE_00 = delta_E_CIE2000(result_cc_mean_lab, self.ideal_lab)
    #     deltaE_76 = colour.delta_E(result_cc_mean_lab, self.ideal_lab, method='CIE 1976')

    #     # return deltaC, deltaE
    #     return deltaC, deltaE_00, deltaE_76

    # def draw_gt_in_image(self, image, image_color_space, deltaE, length=50):
    #     if self.sorted_centroid is None:
    #         self.sorted_centroid, _, _ = detect_colorchecker(image)
    #     image_gt = image.copy()

    #     self.sorted_centroid = np.int32(self.sorted_centroid)
    #     for i in range(len(self.sorted_centroid)):
    #         if image_color_space.lower() == "linear":
    #             image_gt[self.sorted_centroid[i, 1] -
    #                      length:self.sorted_centroid[i, 1] + length,
    #                      self.sorted_centroid[i, 0] -
    #                      length:self.sorted_centroid[i, 0] + length] = lab2rgb(
    #                          self.ideal_lab)[i]
    #         else:
    #             image_gt[self.sorted_centroid[i, 1] -
    #                      length:self.sorted_centroid[i, 1] + length,
    #                      self.sorted_centroid[i, 0] -
    #                      length:self.sorted_centroid[i, 0] + length] = gamma(
    #                          lab2rgb(self.ideal_lab))[i]
    #         cv2.putText(image_gt, str(round(deltaE[i], 1)),
    #                     np.int32(self.sorted_centroid[i]),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)
    #     return image_gt


if __name__ == '__main__':
    pass
