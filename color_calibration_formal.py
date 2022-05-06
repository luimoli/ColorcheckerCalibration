import os
from statistics import mode
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2.cv2 as cv2

from utils.ImageColorCalibration import ImageColorCalibration
from utils import smv_colour
from utils.mcc_detect_color_checker import detect_colorchecker_value
from utils.evaluate_result import evaluate




photo_path = r"./data/mindvision/mvmv_2300.png"
image = cv2.imread(photo_path, -1) / 65535.
cc_mean_value = detect_colorchecker_value(image)
model = ImageColorCalibration(src_for_ccm=cc_mean_value, colorchecker_gt_mode=2)
# model.setCCM_METHOD('polynominal')
model.setColorSpace('linear')
model.setCCM_TYPE('3x3')
model.setCCM_RowSum1(False)
model.run()

print(model.getCCM())

calibratedImage = model.infer(image, image_color_space='linear')
deltaC, deltaE00, img_with_gt = evaluate(calibratedImage, model.ideal_lab, 'linear', 'deltaC')
img_with_gt = np.clip(img_with_gt, 0, 1)
print('deltaC00, deltaE00: ', deltaC.mean(), deltaE00.mean())
cv2.imwrite("img_ccm.png", calibratedImage[..., ::-1]**(1/2.2)*255.)
cv2.imwrite('img_ccm_gt.png', img_with_gt[...,::-1]**(1/2.2)*255.)