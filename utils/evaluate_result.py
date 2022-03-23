import imp
import torch
import numpy as np

from utils import smv_colour
from utils.deltaE.deltaC_2000_np import delta_C_CIE2000
from utils.deltaE.deltaE_2000_np import delta_E_CIE2000
from utils.minimize_ccm import gamma, gamma_reverse

def rgb2lab(data):
    assert data.max() <= 1, "image range should be in [0, 1]"
    data = np.float32(data)
    resultxyz = smv_colour.RGB2XYZ(torch.from_numpy(data), 'bt709')
    resultlab = smv_colour.XYZ2Lab(resultxyz).numpy()
    return resultlab

def evaluate_result(self, result_cc_mean, image_color_space):
    """given an image result, c
    Args:
        result_cc_mean (arr): _description_
        image_color_space (str): ['srgb','linear']
    Returns:
        value: deltaC, deltaE
    """
    assert result_cc_mean.max() <=1 

    if image_color_space == "srgb":
        result_cc_mean_lab = rgb2lab(gamma_reverse(result_cc_mean))
    else:
        result_cc_mean_lab = rgb2lab(result_cc_mean)

    deltaC = delta_C_CIE2000(result_cc_mean_lab, self.ideal_lab)
    deltaE = delta_E_CIE2000(result_cc_mean_lab, self.ideal_lab)
    
    return deltaC, deltaE