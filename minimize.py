import numpy as np
import torch
import smv_colour

from numpy import genfromtxt
from scipy.optimize import minimize
from data import const

def deltaE76(Lab1, Lab2):
    Lab1 = Lab1.cpu().numpy()
    Lab2 = Lab2.cpu().numpy()
    # L1, a1, b1 = Lab1[..., 0], Lab1[..., 1], Lab1[..., 2]
    # L2, a2, b2 = Lab2[..., 0], Lab2[..., 1], Lab2[..., 2]
    # d_E = np.sqrt(np.power(L1-L2) + np.power(a1-a2) + np.power(b1-b2))
    d_E = np.linalg.norm(Lab1 - Lab2, axis=-1)
    return d_E

def fun(realrgb, linear_rgb):
    def v(x):
        
        # realrgb = np.float32(genfromtxt("./data/real_rgb.csv", delimiter=','))
        # nonlinear_rgb = np.float32(genfromtxt("./data/measure_rgb_d65.csv", delimiter=','))
        # realrgb = const.REALRGB
        # linear_rgb = const.NONLINEARRGB ** 2.2


        # ccm = np.array([[1-x1-x2, x1, x2],
        #                 [x3, 1-x3-x4, x4],
        #                 [x5, x6, 1-x5-x6]])

        # ccm = np.array([[1-x[0]-x[1], x[0], x[1]],
        #                 [x[2], 1-x[2]-x[3], x[3]],
        #                 [x[4], x[5], 1-x[4]-x[5]]])

        ccm = np.array([[x[0], x[1], x[2]],
                        [x[3], x[4], x[5]],
                        [x[6], x[7], x[8]]])

        # crgb = np.einsum('...ij,...j->...i', ccm, linear_rgb)
        crgb = np.einsum('ic,hc->hi', const.RGB_XYZ, linear_rgb)

        realrgb_tensor = torch.from_numpy(realrgb)
        realxyz = smv_colour.RGB2XYZ(realrgb_tensor, 'bt709')
        reallab = smv_colour.XYZ2Lab(realxyz)

        crgb_tensor = torch.from_numpy(crgb)
        cxyz = smv_colour.RGB2XYZ(crgb_tensor, 'bt709')
        clab = smv_colour.XYZ2Lab(cxyz)

        deltaE = deltaE76(clab, reallab)
        return deltaE.mean()

    return v
    

if __name__ == '__main__':
    realrgb = const.REALRGB ** 2.2
    linear_rgb = const.NONLINEARRGB ** 2.2
    # args = (realrgb, nonlinear_rgb)  #a
    kr = realrgb[... ,0].mean() / linear_rgb[...,0].mean()
    kg = realrgb[... ,1].mean() / linear_rgb[...,1].mean()
    kb = realrgb[... ,2].mean() / linear_rgb[...,2].mean()

    # x0 = np.float32(np.array([-0.12,-0.44, -0.23,-0.29, 0, -0.54]))  # 初始猜测值
    x0 = np.float32(np.array([kr, 0, 0, 0, kg, 0, 0 ,0 ,kb]))  # 初始猜测值


    res = minimize(fun(realrgb, linear_rgb), x0, method='SLSQP')
    print(res.fun)
    print(res.success)
    print(res.x)