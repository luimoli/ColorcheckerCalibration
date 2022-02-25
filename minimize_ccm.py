import numpy as np
import torch
import cv2
import smv_colour
import pandas as pd 
import scipy.optimize as optimize

from numpy import genfromtxt
from deltaE.deltaE_2000_np import delta_E_CIE2000
# from verify_img_ccm import img_convert_wccm

M_xyz2rgb=np.array([[3.24096994,-1.53738318,-0.49861076],
                    [-0.96924364,1.8759675,0.04155506],
                    [0.05563008,-0.20397695,1.05697151]])
M_rgb2xyz=np.array([[0.4123908 , 0.35758434, 0.18048079],
                    [0.21263901, 0.71516868, 0.07219231],
                    [0.01933082, 0.11919478, 0.95053216]])

def gamma(x,colorspace='sRGB'):
    y=np. zeros (x. shape)
    y[x>1]=1
    if colorspace in ( 'sRGB', 'srgb'):
        y[(x>=0)&(x<=0.0031308)]=(323/25*x[ (x>=0)&(x<=0.0031308)])
        y[(x<=1)&(x>0.0031308)]=(1.055*abs(x[ (x<=1)&(x>0.0031308)])**(1/2.4)-0.055)
    return y

def gamma_reverse(x,colorspace='sRGB'):
    y=np.zeros(x.shape)
    y[x>1]=1
    if colorspace in ('sRGB', 'srgb'):
        y[(x>=0)&(x<=0.04045)]=x[(x>=0)&(x<=0.04045)]/12.92
        y[(x>0.04045)&(x<=1)]=((x[(x>0.04045)&(x<=1)]+0.055)/1.055)**2.4     
    return y

def im2vector(img):
    size=img.shape
    rgb=np.reshape(img,(size[0]*size[1],3))
    func_reverse=lambda rgb : np.reshape(rgb,(size[0],size[1],size[2]))
    return rgb, func_reverse

def ccm(img, ccm):
    if (img.shape[1]==3)&(img.ndim==2):
        rgb=img
        func_reverse=lambda x : x    
    elif (img.shape[2]==3)&(img.ndim==3):
        (rgb,func_reverse)=im2vector(img)    
    rgb=rgb.transpose()
    rgb=ccm@rgb
    rgb=rgb.transpose()    
    img_out=func_reverse(rgb)    
    return img_out

# def rgb2lab(crgb):
#     crgb_tensor = torch.from_numpy(crgb)
#     cxyz = smv_colour.RGB2XYZ(crgb_tensor, 'bt709')
#     clab = smv_colour.XYZ2Lab(cxyz)
#     res_lab = clab.cpu().numpy()
#     return res_lab

def rgb2lab(img, whitepoint='D65'): 
    if (img.ndim==3):
        if (img.shape[2]==3):
            (rgb, func_reverse)=im2vector(img)
    elif (img.ndim==2):
        if (img.shape[1]==3):
            rgb=img
            func_reverse=lambda x : x
        elif (img.shape[0]>80)&(img.shape[1]>80):
            img=np.dstack((img,img,img))
            (rgb, func_reverse)=im2vector(img)
    rgb=rgb.transpose()
    # rgb=gamma_reverse(rgb,colorspace='sRGB')
    xyz=M_rgb2xyz@rgb
    xyz=xyz.transpose()

    xyz_tensor = torch.from_numpy(xyz)
    lab_tensor = smv_colour.XYZ2Lab(xyz_tensor)
    Lab = lab_tensor.cpu().numpy()

    img_out = func_reverse(Lab)
    return img_out


def limit(ccm_matrix, threshold=1.5):
    assert ccm_matrix.shape[0] == ccm_matrix.shape[1] and ccm_matrix.shape[0] == 3
    for i in range(ccm_matrix.shape[0]):
        if ccm_matrix[i][i] > threshold:
            offset = ccm_matrix[i][i] - threshold
            for j in range(ccm_matrix.shape[1]):
                if j == i:
                    ccm_matrix[i][i] = threshold
                else:
                    ccm_matrix[i][j] += (offset/2)
    return ccm_matrix



def ccm_calculate(rgb_data, lab_ideal, mode='default', optimetric='00'):
    """[calculate the color correction matrix]
    Args:
        rgb_data ([N*3]): [the RGB data of color_checker]
        lab_ideal ([N*3]): [the ideal value of color_checker]
        mode (str, optional): [choose how to calculate CCM]. Defaults to 'default'.
                                default: white balance constrain: the sum of row is 1.
                                constrain_1.5: constrain the diagonal value to be less than 1.5 when calculating CCM.
                                limit_1.5: limit CCM's diagonal value after calculation.
    Raises:
        ValueError: [mode value error]
    Returns:
        [array]: [CCM with shape: 3*3]
    """
    x2ccm=lambda x : np.array([[1-x[0]-x[1],x[0],x[1]],
                            [x[2],1-x[2]-x[3],x[3]],
                            [x[4],x[5],1-x[4]-x[5]]])
                   
    # f_lab=lambda x : rgb2lab(gamma(ccm(rgb_data,x2ccm(x)),colorspace='sRGB'))
    f_lab=lambda x : rgb2lab(ccm(rgb_data,x2ccm(x)))

    if optimetric == '76':
        f_error=lambda x : f_lab(x)-lab_ideal
        f_DeltaE=lambda x : np.sqrt((f_error(x)**2).sum(axis=1,keepdims=True)).mean()
    elif optimetric == '00':
        f_DeltaE=lambda x : delta_E_CIE2000(f_lab(x), lab_ideal).mean()
    else:
        raise ValueError(f'optimetric value error!')

    x0=np.array([0,0,0,0,0,0])
    func=lambda x : print('deltaE_00 = ',f_DeltaE(x))

    if mode == 'default':
        result=optimize.minimize(f_DeltaE, x0, method='Powell')
        print('minimize average deltaE00: ', result.fun)
        print(x2ccm(result.x))
        # pd.DataFrame(x2ccm(result.x)).to_csv('ccm.csv',header=False, index=False)
        return x2ccm(result.x)

    elif mode == 'constrain_1.5':
        cons = ({'type': 'ineq', 'fun': lambda x: x[0] + x[1] + 0.5},\
                {'type': 'ineq', 'fun': lambda x: x[2] + x[3] + 0.5},\
                {'type': 'ineq', 'fun': lambda x: x[4] + x[5] + 0.5})
        result=optimize.minimize(f_DeltaE, x0, method='SLSQP', constraints=cons)
        print('minimize average deltaE00: ', result.fun)
        print(x2ccm(result.x))
        return x2ccm(result.x)

    elif mode == 'limit_1.5':
        result=optimize.minimize(f_DeltaE, x0, method='Powell')
        print('minimize average deltaE00: ', result.fun)
        limit_matrix = limit(x2ccm(result.x))
        print(limit_matrix)
        return limit_matrix
        
    else:
        raise ValueError(f'mode value error!')


if __name__ == '__main__':
    # # calculate ccm
    # lab_ideal = np.float32(genfromtxt("./data/real_lab_xrite.csv", delimiter=',')) # from X-rite
    lab_ideal = np.float32(genfromtxt("./data/real_lab.csv", delimiter=',')) # from imatest
    rgb_data = np.float32(genfromtxt("./data/measure_rgb_ck2.csv", delimiter=','))
    ccm_matrix = ccm_calculate(rgb_data, lab_ideal, 'default')

    # image_path = r"./img/colorchecker2.jpg"
    # img_convert_wccm(image_path, ccm_matrix)

