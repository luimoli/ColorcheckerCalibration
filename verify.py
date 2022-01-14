import torch
import smv_colour
import cv2
import numpy as np
import os
import torch
from numpy import genfromtxt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Checker:
    def __init__(self) -> None:
        
        self.D65_to_A_matrix = torch.tensor([[1.2164557,  0.1109905, -0.1549325],
                                                [0.1533326,  0.9152313, -0.0559953],
                                                [-0.0239469,  0.0358984,  0.3147529]], dtype=torch.float32)

        self.A_to_D65_matrix = torch.tensor([[0.8446965, -0.1179225,  0.3948108],
                                                [-0.1366303,  1.1041226,  0.1291718],
                                                [0.0798489, -0.1348999,  3.1924009]], dtype=torch.float32)
        # self.ccm = np.float32(genfromtxt("./data/ccm.csv", delimiter=','))
        # self.ccm = np.array([[ 0.935738, -0.129567, -0.108225],
        #                     [-0.441293,  0.672525,  0.072142],
        #                     [ 0.047443, -0.107284,  0.460978]], dtype=np.float32)

        self.ccm = np.float32(genfromtxt("./data/ccm.csv", delimiter=','))


    def conversion(self, img_rgb):
        img_rgb_linear = img_rgb ** 2.2

        # img_s_xyz = smv_colour.RGB2XYZ(img_rgb_linear, 'bt709')
        # img_d_xyz = smv_colour.dot_vector(self.D65_to_A_matrix, img_s_xyz)
        # img_d_rgb = smv_colour.XYZ2RGB(img_d_xyz, 'bt709')
        
        img_d_rgb = np.einsum('ic,hwc->hwi', self.ccm, img_rgb)

        img_d_rgb = img_d_rgb ** (1 / 1.8)
        
        return img_d_rgb

        # return res


if __name__ == '__main__':
    checker = Checker()


    image_path = r"./img/d65.JPG"
    image_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) / 255.
    image_rgb = np.float32(image_bgr[..., ::-1].copy())
    # image_rgb = torch.from_numpy(image_rgb)
    result_rgb_image = checker.conversion(image_rgb)
    # result_rgb_image = result_rgb_image.cpu().numpy()
    # cv2.imwrite(r"D:\\Code\\VideoHDR-mm\\example-data\\0000002_sdr_no.png", np.uint8(result_rgb_image[..., ::-1] * 255.))
    cv2.imwrite(r"./img/d65_ccm_test.JPG", (result_rgb_image[..., ::-1] * 255.))



    # real_rgb = np.float32(genfromtxt("./data/real_rgb.csv", delimiter=',')) ** 2.2
    # d65_nonlinear_rgb = np.float32(genfromtxt("./data/measure_rgb_d65.csv", delimiter=','))

    # correct_rgb = np.einsum('ic,hc->hi', checker.ccm, d65_nonlinear_rgb)

    # # # has verifed this tranfrom of realrgb -> reallab matches
    # realrgb_tensor = torch.from_numpy(real_rgb)
    # realxyz = smv_colour.RGB2XYZ(realrgb_tensor, 'bt709')
    # reallab = smv_colour.XYZ2Lab(realxyz)

    # crgb_tensor = torch.from_numpy(correct_rgb)
    # cxyz = smv_colour.RGB2XYZ(crgb_tensor, 'bt709')
    # clab = smv_colour.XYZ2Lab(cxyz)

    # print('.')
    # # deltaE = deltaE76(clab, reallab)

