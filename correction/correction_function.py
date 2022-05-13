import cv2.cv2 as cv2
import threading
from PyQt5.QtCore import QFile
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
from PyQt5 import QtGui
import matplotlib.pyplot as plt
import os
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007, demosaicing_CFA_Bayer_bilinear
# import torch
import time
from PyQt5 import QtCore
# from utils import color_science
# from utils import smv_colour
# from utils import mcc_detect_color_checker
# from utils import color_correction
# from utils.color_calibration import ColorCalibration
# from utils import color_calibration
# from utils import mvsdk
from utils import mv_image_func
from utils import smv_colour
from utils import mvsdk
from utils.ImageColorCorrection import ImageColorCorrection
from utils.evaluate_result import evaluate




class CorrectionFunc:
    def __init__(self, ui, mainWnd, result_path, exposure_time, image_format, ccm_type):

        # self.hCamera = None
        # self.FrameBufferSize = None
        # self.pFrameBuffer = None
        # self.init_camera(30)
        self.ui = ui
        self.mainWnd = mainWnd
        self.result_path = result_path
        self.exposure_time = exposure_time
        self.image_format = image_format
        self.ccm_type = ccm_type

        # 默认视频源为相机
        # self.ui.radioButton_AE.setChecked(True)
        self.isCamera = True
        self.raw_image = None
        self.ideal_lab = np.float32(np.loadtxt("E:/code/Color_Calibration_Correction/data/real_lab_imatest.csv",
                                               delimiter=','))

        self.ideal_linear_rgb = smv_colour.XYZ2RGB(smv_colour.Lab2XYZ(self.ideal_lab), 'bt709')

        # 信号槽设置

        self.ui.cc_wb_button.clicked.connect(self.cc_wb_button_function)
        self.ui.wp_wb_button.clicked.connect(self.wp_wb_button_function)
        self.ui.show_image_button.clicked.connect(self.show_image_button_function)
        self.ui.exposure_button.clicked.connect(self.exposure_button_function)
        self.ui.save_image_button.clicked.connect(self.save_image_button_function)
        self.ui.correction_button.clicked.connect(self.correction_button_function)
        self.ui.multilight_wp_wb_button.clicked.connect(self.multilight_wp_wb_button_function)
        self.ui.multilight_correction_button.clicked.connect(self.multilight_correction_button_function)

        self.hCamera = None
        self.FrameBufferSize = None
        self.pFrameBuffer = None
        self.hCamera = mv_image_func.init_camera(1, exposure_time)

        self.stopEvent = threading.Event()
        self.stopEvent.clear()
        self.play = True
        self.FrameHead = None

        self.calib_image_num = 10
        self.rgb_gain = np.array([1, 1, 1])
        self.cct = None

        self.multilight_wb_gain = None
        self.multilight_alpha = None

        self.white_point = np.array([1, 1, 1])
        self.cct_ccm_dict = np.load(result_path, allow_pickle=True).item()

        self.ui.raw_image_viewer.setGeometry(QtCore.QRect(self.ui.raw_image_viewer.geometry().x(),
                                                          self.ui.raw_image_viewer.geometry().y(),
                                                          2448//8, 2048//8))

        cct_ccm_dict = np.load(result_path, allow_pickle=True).item()
        self.correction_model = ImageColorCorrection(cct_ccm_dict, "linear")
        # image_color_correction.setMethod("cc")
        # image_color_correction.doWhiteBalance(wb_image=image_wb)
        # corrected_image = image_color_correction.correctImage(image, "linear")


    def show_image_button_function(self):
        th = threading.Thread(target=self.Display)
        th.start()

    def Display(self):
        while True:
            frame = mv_image_func.get_image(self.hCamera, self.image_format)
            # frame = self.get_image()
            assert self.image_format.lower() in ["png", "jpg", "jpeg", "tiff", "bmp", "raw"], "-------"
            if self.image_format.lower() in ["png", "jpg", "jpeg", "tiff", "bmp"]:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            elif self.image_format.lower() == "raw":
                img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_Grayscale16)

            pix_img = QPixmap.fromImage(img)
            # pix_img = pix_img.scaled(300, 300, QtCore.Qt.KeepAspectRatio)
            # pix_img = pix_img.scaled(frame.shape[1] // 8, frame.shape[0] // 8, QtCore.Qt.KeepAspectRatio)
            self.ui.raw_image_viewer.setPixmap(pix_img)

            cv2.waitKey(1)
            # 判断关闭事件是否已触发
            if True == self.stopEvent.is_set():
                # 关闭事件置为未触发，清空显示label
                self.stopEvent.clear()
                self.ui.raw_image_viewer.clear()
                break

    def show_image_in_Qlabel(self, image, viewer):
        img3 = QtGui.QImage(image, image.shape[1], image.shape[0], image.shape[1] * 3, QtGui.QImage.Format_RGB888)
        image2 = QtGui.QPixmap(img3).scaled(viewer.width(), viewer.height())
        viewer.setPixmap(image2)
        viewer.setScaledContents(True)

    def exposure_button_function(self):
        value_int = int(self.ui.exposure_value.text())
        message = self.ui.exposure_comboBox.currentText()
        if "auto" in message.lower():
            method = 1
        else:
            method = 0

        self.stopEvent.set()
        mvsdk.CameraUnInit(self.hCamera)

        # 释放帧缓存
        mvsdk.CameraAlignFree(self.pFrameBuffer)
        print(message, value_int)
        self.hCamera = mv_image_func.init_camera(method, value_int)
        self.stopEvent.clear()
        self.show_image_button_function()

    def save_image_button_function(self):
        raw_image = mv_image_func.get_image(self.hCamera, self.image_format, 1)
        cv2.imwrite("save_image_%f.png" % time.time(), np.uint16(raw_image*65535))
        return

    def cc_wb_button_function(self):
        image = mv_image_func.get_rgb_image(self.hCamera, self.image_format)
        self.correction_model.setMethod("cc")
        self.correction_model.doWhiteBalance(image)
        return

    def multilight_wp_wb_button_function(self):
        image = mv_image_func.get_rgb_image(self.hCamera, self.image_format)
        self.correction_model.setMethod("multiple_light")
        self.correction_model.doWhiteBalance(image)
        return

    def wp_wb_button_function(self):
        image = mv_image_func.get_rgb_image(self.hCamera, self.image_format)
        self.correction_model.setMethod("wp")
        self.correction_model.doWhiteBalance(image)
        # cct = self.correction_model.cct
        return

    def correction_button_function(self):
        image = mv_image_func.get_rgb_image(self.hCamera, self.image_format)
        corrected_image = self.correction_model.correctImage(image, "linear")
        deltaC, deltaE00, img_with_gt = evaluate(corrected_image, self.ideal_lab, 'linear', 'deltaC')
        print(deltaC.mean(), deltaE00.mean())
        self.show_image_in_Qlabel((img_with_gt ** (1/2.2) * 255).astype(np.uint8), self.ui.result_viewer)
        cv2.imwrite("img_with_gt.png", img_with_gt[..., ::-1]*255)
        return

    def multilight_correction_button_function(self):
        image = mv_image_func.get_rgb_image(self.hCamera, self.image_format)
        corrected_image = self.correction_model.multiLightCorrectImage(image, "linear")
        deltaC, deltaE00, img_with_gt = evaluate(corrected_image, self.ideal_lab, 'linear', 'deltaC')
        print(deltaC.mean(), deltaE00.mean())
        self.show_image_in_Qlabel((img_with_gt ** (1 / 2.2) * 255).astype(np.uint8), self.ui.result_viewer)
        cv2.imwrite("img_with_gt.png", img_with_gt[..., ::-1] * 255)
        return
