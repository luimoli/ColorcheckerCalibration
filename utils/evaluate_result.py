import numpy as np
import cv2
import os,sys
sys.path.append(os.path.abspath('../'))

from utils.deltaE.deltaC_2000_np import delta_C_CIE2000
from utils.deltaE.deltaE_2000_np import delta_E_CIE2000
from utils.misc import gamma, gamma_reverse, rgb2lab, lab2rgb, deltaE76, deltaC76
from utils import mcc_detect_color_checker


def evaluate(image, ideal_lab, image_color_space, draw_mode='deltaC'):
    """evaluate the image with colorchecker.

    Args:
        image (arr): _description_
        ideal_lab (arr): gt
        image_color_space (str): ["srgb", 'linear']
        draw_mode (str, optional): ['deltaC', 'deltaE']. Defaults to 'deltaC'.

    Returns:
        list: deltaC, deltaE_00, image_with_gt
    """
    sorted_centroid, length, result_cc_mean, _ = mcc_detect_color_checker.detect_colorchecker(image)

    result_cc_mean = np.clip(result_cc_mean, 0, 1)
    if image_color_space == "srgb":
        result_cc_mean_lab = rgb2lab(gamma_reverse(result_cc_mean))
    elif image_color_space == "linear":
        result_cc_mean_lab = rgb2lab(result_cc_mean)
    else:
        raise ValueError(image_color_space)
    deltaC = delta_C_CIE2000(result_cc_mean_lab, ideal_lab)
    deltaE_00 = delta_E_CIE2000(result_cc_mean_lab, ideal_lab)
    deltaE_76 = deltaE76(result_cc_mean_lab, ideal_lab)
    deltaC_76 = deltaC76(result_cc_mean_lab, ideal_lab)

    if draw_mode == 'deltaC':
        delta_draw = deltaC.copy()
    elif draw_mode == 'deltaE':
        delta_draw == deltaE_00.copy()
    else:
        raise ValueError(draw_mode)

    image_with_gt = draw_gt_in_image(image, image_color_space, ideal_lab, delta_draw, sorted_centroid, length)

    return deltaC, deltaE_00, image_with_gt
    # return deltaC, deltaE_00, image_with_gt, deltaE_76, deltaC_76



def draw_gt_in_image(image, image_color_space, ideal_lab, deltaE, sorted_centroid, length):
    image_gt = image.copy()
    for i in range(len(sorted_centroid)):
        if image_color_space.lower() == "linear":
            image_gt[sorted_centroid[i, 1] - length:sorted_centroid[i, 1] +
                     length,
                     sorted_centroid[i, 0] - length:sorted_centroid[i, 0] +
                     length] = lab2rgb(ideal_lab)[i]
        else:
            image_gt[sorted_centroid[i, 1] - length:sorted_centroid[i, 1] +
                     length,
                     sorted_centroid[i, 0] - length:sorted_centroid[i, 0] +
                     length] = gamma(lab2rgb(ideal_lab))[i]
            
        cv2.putText(image_gt, str(round(deltaE[i], 1)),
                    np.int32(sorted_centroid[i]), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 255, 0), 2)
    return image_gt

if __name__ == '__main__':
    image_ = cv2.imread(r"../data/tmp/2022-06-08_16-06-38_857.jpg")[..., ::-1] / 255.
    ideal_lab = np.float32(np.loadtxt(r'../data/real_lab_xrite.csv', delimiter=','))
    deltaC, deltaE00, image_with_gt = evaluate(image_, ideal_lab, image_color_space='linear', draw_mode='deltaC')
    image_with_gt = np.clip(image_with_gt, 0, 1)
    cv2.imwrite(r'../data/tmp/2022-06-08_16-06-38_857_gt.png', image_with_gt[...,::-1]**(1/2.2)*255.)
    cv2.imwrite(r'../data/tmp/2022-06-08_16-06-38_857_gamma.png', image_[...,::-1]**(1/2.2)*255.)

    print('deltaC, deltaE00:  ', deltaC.mean(), deltaE00.mean())