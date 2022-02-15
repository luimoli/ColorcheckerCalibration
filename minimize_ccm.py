import numpy as np
import torch
import cv2
import smv_colour
import pandas as pd 
import scipy.optimize as optimize

from numpy import genfromtxt
from scipy.optimize import minimize
from deltaE.deltaE_2000_np import delta_E_CIE2000

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

def rgb2lab(img,whitepoint='D65'): 
    if (img.ndim==3):
        if (img.shape[2]==3):
            (rgb,func_reverse)=im2vector(img)
    elif (img.ndim==2):
        if (img.shape[1]==3):
            rgb=img
            func_reverse=lambda x : x
        elif (img.shape[0]>80)&(img.shape[1]>80):
            img=np.dstack((img,img,img))
            (rgb,func_reverse)=im2vector(img)
    rgb=rgb.transpose()
    # rgb=gamma_reverse(rgb,colorspace='sRGB')
    xyz=M_rgb2xyz@rgb
    xyz=xyz.transpose()

    xyz_tensor = torch.from_numpy(xyz)
    lab_tensor = smv_colour.XYZ2Lab(xyz_tensor)
    Lab = lab_tensor.cpu().numpy()

    # f=lambda t : (t>((6/29)**3))*(t**(1/3))+\
    #     (t<=(6/29)**3)*(29*29/6/6/3*t+4/29)
    # if whitepoint=='D65':
    #     Xn = 95.047/100
    #     Yn = 100/100
    #     Zn = 108.883/100
    # L = 116 * f(xyz[:,1]/Yn) - 16
    # a = 500 * (f(xyz[:,0]/Xn) - f(xyz[:,1]/Yn))
    # b = 200 * (f(xyz[:,1]/Yn) - f(xyz[:,2]/Zn))
    # Lab = np.vstack((L,a,b)).transpose()

    img_out = func_reverse(Lab)
    return img_out

def ccm_calculate(rgb_data, lab_ideal):
    """[summary]
    Args:
        rgb_data ([N*3]): [the RGB data of color_checker]
        lab_ideal ([N*3]): [the ideal value of color_checker]
    Returns:
        [nparray]: [shape: 3*3]
    """
    x2ccm=lambda x : np.array([[1-x[0]-x[1],x[0],x[1]],
                            [x[2],1-x[2]-x[3],x[3]],
                            [x[4],x[5],1-x[4]-x[5]]])
                   
    # f_lab=lambda x : rgb2lab(gamma(ccm(rgb_data,x2ccm(x)),colorspace='sRGB'))
    f_lab=lambda x : rgb2lab(ccm(rgb_data,x2ccm(x)))

    # # --- deltaE 76
    # f_error=lambda x : f_lab(x)-lab_ideal
    # f_DeltaE=lambda x : np.sqrt((f_error(x)**2).sum(axis=1,keepdims=True)).mean()

    # # --- deltaE 00
    f_DeltaE=lambda x : delta_E_CIE2000(f_lab(x), lab_ideal).mean()

    x0=np.array([0,0,0,0,0,0])
    func=lambda x : print('x = ',f_DeltaE(x))
    result=optimize.minimize(f_DeltaE,x0,method='Powell',callback=func)

    print(result)
    print(x2ccm(result.x))
    pd.DataFrame(x2ccm(result.x)).to_csv('ccm.csv',header=False, index=False)
    return x2ccm(result.x)

def conversion(img_rgb, ccm):
    h, w, c = img_rgb.shape
    img_rgb_reshape = np.reshape(img_rgb,(h*w, c))
    # ccres = img_rgb_reshape.mm(ccm)
    ccres = np.matmul(img_rgb_reshape, ccm)
    img_d_rgb = np.reshape(ccres, (h,w,c))
    img_d_rgb[img_d_rgb > 1] = 1
    img_d_rgb[img_d_rgb < 0] = 0

    img_d_rgb = gamma(img_d_rgb,colorspace='sRGB')
    # img_d_rgb = img_d_rgb ** (1 / 2.2)

    img_d_rgb[img_rgb == 1] = 1
    return img_d_rgb

if __name__ == '__main__':
    # # calculate ccm
    # lab_ideal = np.float32(genfromtxt("./data/real_lab_xrite.csv", delimiter=',')) # from X-rite
    lab_ideal = np.float32(genfromtxt("./data/real_lab.csv", delimiter=',')) # from imatest
    rgb_data = np.float32(genfromtxt("./data/measure_rgb_ck2.csv", delimiter=','))

    ccm_matrix = ccm_calculate(rgb_data, lab_ideal)

    # # convert img with ccm
    image_path = r"./img/colorchecker2.jpg"
    image_bgr = cv2.imread(image_path)  / 255.
    image_rgb = np.float32(image_bgr[..., ::-1].copy())
    result_rgb_image = conversion(image_rgb, ccm_matrix)
    cv2.imwrite(image_path[:-4]+'_' + 'ccm_minimize_2_test'+'.jpg', (result_rgb_image[..., ::-1] * 255.))
