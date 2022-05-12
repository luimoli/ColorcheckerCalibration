import numpy as np
import cv2

from utils import smv_colour
from utils.deltaE.deltaC_2000_np import delta_C_CIE2000
from utils.deltaE.deltaE_2000_np import delta_E_CIE2000
from utils.misc import gamma, gamma_reverse, rgb2lab, lab2rgb
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
    result_cc_mean = mcc_detect_color_checker.detect_colorchecker_value(image)
    result_cc_mean = np.clip(result_cc_mean, 0, 1)

    if image_color_space == "srgb":
        result_cc_mean_lab = rgb2lab(gamma_reverse(result_cc_mean))
    elif image_color_space == "linear":
        result_cc_mean_lab = rgb2lab(result_cc_mean)
    else:
        raise ValueError(image_color_space)

    deltaC = delta_C_CIE2000(result_cc_mean_lab, ideal_lab)
    deltaE_00 = delta_E_CIE2000(result_cc_mean_lab, ideal_lab)
    # deltaE_76 = colour.delta_E(result_cc_mean_lab, ideal_lab, method='CIE 1976')

    if draw_mode == 'deltaC':
        delta_draw = deltaC.copy()
    elif draw_mode == 'deltaE':
        delta_draw == deltaE_00.copy()
    else:
        raise ValueError(draw_mode)

    image_with_gt = draw_gt_in_image(image, image_color_space, ideal_lab, delta_draw)

    return deltaC, deltaE_00, image_with_gt


def draw_gt_in_image(image, image_color_space, ideal_lab, deltaE, length=50):
    sorted_centroid, _, _ = mcc_detect_color_checker.detect_colorchecker(image)
    image_gt = image.copy()
    sorted_centroid = np.int32(sorted_centroid)
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
    image_ = cv2.imread(r"./data/tmp/image_D65.bmp")[..., ::-1] / 255.
    ideal_lab = None
    deltaC, deltaE00, image_with_gt = evaluate(image_, ideal_lab, image_color_space='srgb', draw_mode='deltaC')
    # image_with_gt = icc_minimize.draw_gt_in_image(image_, "linear", deltaC)
    image_with_gt = np.clip(image_with_gt, 0, 1)
    cv2.imwrite(r'./data/tmp/image_D65_gt.bmp', image_with_gt[...,::-1]*255.)
    print('deltaC, deltaE00:  ', deltaC.mean(), deltaE00.mean())