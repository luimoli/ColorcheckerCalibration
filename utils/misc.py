import numpy as np
import torch

from utils import smv_colour

def gamma(x, colorspace='sRGB'):
    y = np.zeros(x.shape)
    y[x > 1] = 1
    if colorspace in ('sRGB', 'srgb'):
        y[(x >= 0) & (x <= 0.0031308)] = (323 / 25 *
                                          x[(x >= 0) & (x <= 0.0031308)])
        y[(x <= 1) & (x > 0.0031308)] = (
            1.055 * abs(x[(x <= 1) & (x > 0.0031308)])**(1 / 2.4) - 0.055)
    return y


def gamma_reverse(x, colorspace='sRGB'):
    y = np.zeros(x.shape)
    y[x > 1] = 1
    if colorspace in ('sRGB', 'srgb'):
        y[(x >= 0) & (x <= 0.04045)] = x[(x >= 0) & (x <= 0.04045)] / 12.92
        y[(x > 0.04045) & (x <= 1)] = ((x[(x > 0.04045) & (x <= 1)] + 0.055) /
                                       1.055)**2.4
    return y

def deltaE76(Lab1, Lab2):
    d_E = np.linalg.norm(Lab1 - Lab2, axis=-1)
    return d_E

def rgb2lab(data):
    data = np.clip(data, 0, 1) #TODO
    # assert data.max() <= 1, "image range should be in [0, 1]"
    data = np.float32(data)
    resultxyz = smv_colour.RGB2XYZ(torch.from_numpy(data), 'bt709')
    resultlab = smv_colour.XYZ2Lab(resultxyz).numpy()
    return resultlab

def lab2rgb(data):
    resultxyz = smv_colour.Lab2XYZ(torch.from_numpy(np.float32(data)))
    resultrgb = smv_colour.XYZ2RGB(resultxyz, 'bt709').numpy()
    return resultrgb