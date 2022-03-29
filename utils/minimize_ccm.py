import numpy as np
import torch
from utils import smv_colour
import scipy.optimize as optimize
import cv2.cv2 as cv2
from utils import detect_color_checker
from numpy import genfromtxt
from utils.deltaE.deltaE_2000_np import delta_E_CIE2000
import colour
# from verify_img_ccm import img_convert_wccm


M_xyz2rgb = np.array([[3.24096994, -1.53738318, -0.49861076],
                      [-0.96924364, 1.8759675, 0.04155506],
                      [0.05563008, -0.20397695, 1.05697151]])
M_rgb2xyz = np.array([[0.4123908, 0.35758434, 0.18048079],
                      [0.21263901, 0.71516868, 0.07219231],
                      [0.01933082, 0.11919478, 0.95053216]])


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


def im2vector(img):
    size = img.shape
    rgb = np.reshape(img, (size[0] * size[1], 3))
    func_reverse = lambda rgb: np.reshape(rgb, (size[0], size[1], size[2]))
    return rgb, func_reverse


def ccm(img, ccm):
    # if (img.shape[1] == 3) & (img.ndim == 2):
    #     rgb = img
    #     func_reverse = lambda x: x
    # elif (img.shape[2] == 3) & (img.ndim == 3):
    #     (rgb, func_reverse) = im2vector(img)
    rgb = img.transpose()
    rgb = ccm @ rgb
    rgb = rgb.transpose()
    # img_out = func_reverse(rgb)
    return rgb


def rgb2lab(img):
    if (img.ndim == 3):
        if (img.shape[2] == 3):
            (rgb, func_reverse) = im2vector(img)
    elif (img.ndim == 2):
        if (img.shape[1] == 3):
            rgb = img
            func_reverse = lambda x: x
        elif (img.shape[0] > 80) & (img.shape[1] > 80):
            img = np.dstack((img, img, img))
            (rgb, func_reverse) = im2vector(img)
    rgb = rgb.transpose()
    xyz = M_rgb2xyz @ rgb
    xyz = xyz.transpose()
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




def ccm_calculate(rgb_data, lab_ideal, ccm_weight, ccm_space="linear", mode='default_wo_row', optimetric='00'):
    """[calculate the color correction matrix]

    Args:
        rgb_data ([N*3]): [the RGB data of color_checker]
        lab_ideal ([N*3]): [the ideal value of color_checker]
        ccm_space (str, optional): ['srgb', 'linear']. Defaults to "linear".
        mode (str, optional): [choose how to calculate CCM]. Defaults to 'default'.
                                default: white balance constrain: the sum of row is 1.
                                constrain_1.5: constrain the diagonal value to be less than 1.5 when calculating CCM.
                                limit_1.5: limit CCM's diagonal value after calculation.
                                default_wo_row: 
        optimetric (str, optional): the metric of CCM optimization:['00', 76']. Defaults to '00'.
    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
    Returns:
        [array]: [CCM with shape: 3*3]
    """
    if rgb_data.shape[1] == 3:
        if mode == 'default_wo_row':
            x2ccm=lambda x : np.array([[x[0], x[1],x[2]],
                                      [x[3], x[4], x[5]],
                                      [x[6],x[7],x[8]]])
        else:     
            x2ccm=lambda x : np.array([[1-x[0]-x[1],x[0],x[1]],
                                    [x[2],1-x[2]-x[3],x[3]],
                                    [x[4],x[5],1-x[4]-x[5]]])
    else:
        x2ccm=lambda x : np.array([[x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]],
                            [x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17]],
                            [x[18], x[19], x[20], x[21], x[22], x[23], x[24], x[25], x[26]]])


    if ccm_space == "linear":
        f_lab=lambda x : rgb2lab(ccm(rgb_data, x2ccm(x)))
    elif ccm_space.lower() == "srgb":
        f_lab = lambda x: rgb2lab(gamma_reverse(ccm(rgb_data, x2ccm(x)), colorspace='sRGB'))
    else:
        raise ValueError(f'ccm_space value error!')


    if optimetric == '76':
        f_error=lambda x : f_lab(x)-lab_ideal
        # f_DeltaE=lambda x : np.sqrt((f_error(x)**2).sum(axis=1,keepdims=True)).mean()
        f_DeltaE=lambda x : (np.sqrt((f_error(x)**2).sum(axis=1)) * ccm_weight).mean()
    elif optimetric == '00':
        # f_DeltaE=lambda x : (delta_E_CIE2000(f_lab(x), lab_ideal) * ccm_weight).mean()
        f_DeltaE=lambda x : ((delta_E_CIE2000(f_lab(x), lab_ideal) * ccm_weight)**2).mean()
    else:
        raise ValueError(f'optimetric value error!')

    if rgb_data.shape[1] == 3:
        if mode == 'default_wo_row':
            # x0 = np.array([1,0,0, 0,1,0, 0,0,1])
            x0 = np.zeros((9))
            # x0 = np.ones((9))
        else:
            x0=np.array([0,0,0,0,0,0])
    else:
        x0=np.zeros((27))

    func=lambda x : print('deltaE_00 = ',f_DeltaE(x))

    if mode == 'default' or mode == 'default_wo_row':
        result=optimize.minimize(f_DeltaE, x0, callback=func, method='Powell')
        print('minimize average deltaE00: ', result.fun)
        print(x2ccm(result.x))
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


def compute_mean_value(image, sorted_centroid, length):
    sorted_centroid2 = np.int32(sorted_centroid)
    mean_value = np.empty((sorted_centroid.shape[0], 3))
    for i in range(len(sorted_centroid)):
        mean_value[i] = np.mean(image[sorted_centroid2[i, 1] - length:sorted_centroid2[i, 1] + length,
                                sorted_centroid2[i, 0] - length:sorted_centroid2[i, 0] + length], axis=(0, 1))
    return mean_value


def deltaE76(Lab1, Lab2):
    d_E = np.linalg.norm(Lab1 - Lab2, axis=-1)
    return d_E


if __name__ == "__main__":
    lab_ideal = np.loadtxt("../data/real_rgb.csv", delimiter=",")
    # image = cv2.imread(r"E:\data\mindvision\d65\d65_colorchecker.jpg")[..., ::-1] / 255.
    image = cv2.imread(r"E:\data\mindvision\A_light\exposure30.jpg")[..., ::-1] / 255.
    # image = cv2.imread(r"C:\Users\30880\Desktop\nikon_Alight\DSC_4263.JPG")
    # image = cv2.resize(image, (1768, 1179))[..., ::-1] / 255.


    # print(image.shape)
    # exit()


    # image = cv2.imread(r"E:\data\mindvision\20220217data\A\A_colorchecker.png", cv2.IMREAD_UNCHANGED)[..., ::-1] / 65535.
    sorted_centroid, clusters = detect_color_checker.detect_color_checker(image)
    mean_value = compute_mean_value(image, sorted_centroid, 50)
    gain = np.max(mean_value[18:], axis=1)[:, None] / mean_value[18:, ]
    rgb_gain = gain[0:3].mean(axis=0)
    print("rgb_gain:", rgb_gain)
    print("white point:", 1 / rgb_gain)
    # exit()
    image_WB = image * rgb_gain[None, None]

    image_WB_mean = compute_mean_value(image_WB, sorted_centroid, 50)
    ill_gain = (lab_ideal / image_WB_mean)
    image_WB_ill = image_WB * ill_gain[18:21].mean()
    image_WB_mean_ill_mean = compute_mean_value(image_WB_ill, sorted_centroid, 50)
    # cv2.imwrite("A_colorchecker_WB.jpg", np.uint8(image_WB_ill[..., ::-1]))


    lab_ideal = np.float32(genfromtxt("../data/real_lab.csv", delimiter=',')) # from imatest
    rgb_data = np.float32(genfromtxt("./data/measure_rgb_ck2.csv", delimiter=','))
    ccm_matrix = ccm_calculate(image_WB_mean_ill_mean, lab_ideal)
    result_image = np.einsum('ic, hwc->hwi', ccm_matrix, image_WB_ill)
    result_image_mean = compute_mean_value(result_image, sorted_centroid, 50)
    result_image_mean = np.float32(result_image_mean)

    print("ccm_matrix:", ccm_matrix)

    resultxyz = smv_colour.RGB2XYZ(torch.from_numpy(result_image_mean), 'bt709')
    resultlab = smv_colour.XYZ2Lab(resultxyz).numpy()
    print(colour.delta_E(resultlab, lab_ideal, 'CIE 2000').mean())
    # print(deltaE76(resultlab, lab_ideal).mean())
    # cv2.imwrite("nikon_Alight.png", np.uint8(result_image[..., ::-1]**(1/2.2)*255))
    # plt.figure()
    # plt.imshow(image)
    # plt.figure()
    # plt.imshow(image_WB)
    # plt.figure()
    # plt.imshow((image_WB_ill))
    # plt.figure()
    # plt.imshow((result_image)**(1/2.2))
    # plt.show()


# image_path = r"./img/colorchecker2.jpg"
# img_convert_wccm(image_path, ccm_matrix)
