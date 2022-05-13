import sys
sys.path.append("..")
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
import serial
from PyQt5 import QtCore
# from utils import color_science
# from utils import color_calibration
# from utils import smv_colour
# from utils import mcc_detect_color_checker
# from utils import color_correction
# from utils.color_calibration import ColorCalibration
from utils import mvsdk
from utils import mv_image_func
from utils import mcc_detect_color_checker
from utils.ImageColorCalibration import ImageColorCalibration
from utils.evaluate_result import evaluate
from utils.misc import gamma, gamma_reverse

class CalibrationFunc:
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
        # self.ideal_lab = np.float32(np.loadtxt("E:/code/Color_Calibration_Correction/data/real_lab_imatest.csv",
        #                                        delimiter=','))

        # self.ideal_lab = np.float32(color_calibration.ColorChecker2005_3nh)
        # self.ideal_linear_rgb = smv_colour.XYZ2RGB(smv_colour.Lab2XYZ(torch.from_numpy(self.ideal_lab)), 'bt709').numpy()

        # 信号槽设置

        # myUi.cc_detect_button.clicked.connect(lambda: cc_detect_button_function(image, myUi.result_viewer))
        self.ui.cc_detect_button.clicked.connect(self.cc_detect_button_function)
        self.ui.calibration_button.clicked.connect(self.calibration_button_function)
        self.ui.exposure_button.clicked.connect(self.exposure_button_function)
        # self.ui.save_image_button.clicked.connect(self.save_image_button_function)
        self.ui.quick_calibration_button.clicked.connect(self.quick_calibration_button_function)

        self.ui.show_image_button.clicked.connect(self.show_image_button_function)


        # self.hCamera = None
        self.FrameBufferSize = None
        self.pFrameBuffer = None
        self.hCamera = mv_image_func.init_camera(1, exposure_time)

        self.stopEvent = threading.Event()
        self.stopEvent.clear()
        self.play = True
        self.FrameHead = None
        self.calib_image_num = 10
        # ser.close()

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

    def cc_detect_button_function(self):
        image = mv_image_func.get_rgb_image(self.hCamera, self.image_format)
        sorted_centroid, length, charts_RGB, marker_image = mcc_detect_color_checker.detect_colorchecker(image)
        marker_image = (marker_image * 255).astype(np.uint8)
        self.show_image_in_Qlabel(marker_image, self.ui.result_viewer)
        return

    def calibration_button_function(self):
        image = mv_image_func.get_rgb_image(self.hCamera, self.image_format, self.calib_image_num)
        print("get_rgb_image finished~~~~")
        # image = np.load(r"E:\code\PyQt_practice\raw_image1.npy")
        _, _, cc_mean_value, _ = mcc_detect_color_checker.detect_colorchecker(image)
        model = ImageColorCalibration(src_for_ccm=cc_mean_value, colorchecker_gt_mode=1)
        model.setCCM_METHOD('minimize')
        model.setColorSpace('linear')
        model.setCCM_TYPE('3x4')
        model.setCCM_RowSum1(False)
        model.run()
        ccm = model.getCCM()
        model.save(self.result_path)

        calibratedImage = model.infer(image, image_color_space='linear', illumination_gain=False, ccm_correction=True)
        deltaC, deltaE00, img_with_gt = evaluate(calibratedImage, model.ideal_lab, 'linear', 'deltaC')
        img_with_gt = np.clip(img_with_gt, 0, 1)
        self.show_image_in_Qlabel((img_with_gt * 255).astype(np.uint8), self.ui.result_viewer)
        cv2.imwrite("img_with_gt.png", gamma(img_with_gt[..., ::-1])*255)
        print(deltaC.mean(), deltaC.max())
        print(deltaE00.mean(), deltaE00.max())
        return

    def quick_calibration_button_function(self):
        portx = "COM3"
        self.ser = serial.Serial(portx, 9600, timeout=5)
        # result = self.ser.write("<1,S=OFF>".encode("gbk"))

        self.ser.write("<1,S=ON>".encode("gbk"))
        for color_temp in range(0, 101, 10):
            time.sleep(1)
            self.ser.write(("<1,C=%d>" % color_temp).encode("gbk"))
            time.sleep(3)
            self.calibration_button_function()

        self.ser.write("<1,S=OFF>".encode("gbk"))
        return

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