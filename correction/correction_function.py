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
        # self.ui.multilight_wp_wb_button.clicked.connect(self.multilight_wp_wb_button_function)
        # self.ui.multilight_correction_button.clicked.connect(self.multilight_correction_button_function)

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

    def wp_wb_button_function(self):
        image = mv_image_func.get_rgb_image(self.hCamera, self.image_format)
        self.correction_model.setMethod("wp")
        self.correction_model.doWhiteBalance(image)
        return

    def correction_button_function(self):
        image = mv_image_func.get_rgb_image(self.hCamera, self.image_format)
        corrected_image = self.correction_model.correctImage(image, "linear")
        deltaC, deltaE00, img_with_gt = evaluate(corrected_image, self.ideal_lab, 'linear', 'deltaC')
        print(deltaC.mean(), deltaE00.mean())
        self.show_image_in_Qlabel((img_with_gt ** (1/2.2) * 255).astype(np.uint8), self.ui.result_viewer)
        cv2.imwrite("img_with_gt.png", img_with_gt[..., ::-1]*255)
        return

'''
    def mouseMoveEvent(self, event):
        print(event.x(), event.y())

    def raw_image_isp(self, raw_image):
        print("raw_image_isp begin!!!")
        raw_image = color_calibration.read_raw_data(raw_image, self.FrameHead.iHeight, self.FrameHead.iWidth)
        raw_image = demosaicing_CFA_Bayer_bilinear(raw_image, pattern="RGGB")[..., ::-1]
        print("raw_image_isp finished!!!")

        return np.float32(raw_image)

    def correction_button_function(self):
        cct = color_science.calculate_cct(self.white_point)
        ccm = np.array(color_correction.predict_ccm(self.cct_ccm_dict, cct))
        print(cct, ccm)
        self.raw_image = self.get_image()
        if self.image_format.lower() in ["raw"]:
            self.raw_image = self.raw_image_isp(self.raw_image)
        else:
            self.raw_image = self.raw_image / 255.


        image_calib = color_correction.image_correction(self.raw_image, "linear",
                                                        np.array([self.R_Gain, self.G_Gain, self.B_Gain]),
                                                        1, ccm, True, False, True)
        sorted_centroid, charts_RGB, marker_image = mcc_detect_color_checker.detect_color_checker(self.raw_image)
        deltaC, deltaE = color_correction.evaluate_result(image_calib, "linear", sorted_centroid, self.ideal_lab)
        image_gt = color_correction.draw_gt_in_image(image_calib, "linear", deltaC, sorted_centroid, self.ideal_linear_rgb)
        image_calib = np.clip(image_calib, 0, 1)
        self.show_image_in_Qlabel(np.uint8(image_calib ** (1/2.2) * 255), self.ui.result_viewer)

        print(deltaC.mean(), deltaE.mean())
        print(deltaC.mean(), deltaC.max())
        print(deltaE.mean(), deltaE.max())
        cv2.imwrite("./data/image_gt_%f.png"%time.time(), np.uint8(image_calib[..., ::-1] ** (1/2.2) * 255))
        # cv2.imwrite("raw_image.png", np.uint8(self.raw_image*255))

        # calib_image = color_correction.image_correction(self.raw_image, )

        return

    def multilight_correction_button_function(self):
        self.raw_image = self.get_image()
        if self.image_format.lower() in ["raw"]:
            self.raw_image = self.raw_image_isp(self.raw_image)
        else:
            self.raw_image = self.raw_image / 255.

        image_ori_wb = self.raw_image * self.multilight_wb_gain

        cct1 = 4739.57958984375
        cct2 = 6502.56591796875
        ccm1 = self.cct_ccm_dict[cct1]
        ccm2 = self.cct_ccm_dict[cct2]
        image_wb_ccm1 = color_correction.image_correction(image_ori_wb, "linear", None, None, ccm1, False, False, True)
        image_wb_ccm2 = color_correction.image_correction(image_ori_wb, "linear", None, None, ccm2, False, False, True)

        image_calib = self.multilight_alpha * image_wb_ccm1 + (1 - self.multilight_alpha) * image_wb_ccm2

        image_calib = np.clip(image_calib, 0, 1)
        self.show_image_in_Qlabel(np.uint8(image_calib ** (1/2.2) * 255), self.ui.result_viewer)

        cv2.imwrite("image_gt_%f.png"%time.time(), np.uint8(image_calib[..., ::-1] ** (1/2.2) * 255))

        return

    def wp_wb_button_function(self):
        self.raw_image = self.get_image()
        if self.image_format.lower() in ["raw"]:
            self.raw_image = self.raw_image_isp(self.raw_image)
        else:
            self.raw_image = self.raw_image / 255.

        height, width = self.raw_image.shape[0], self.raw_image.shape[1]
        wp_mean = np.mean(self.raw_image[int(height*3/8):int(height*5/8), int(width*3/8):int(width*5/8)], axis=(0, 1))
        self.R_Gain = wp_mean.max() / wp_mean[0] * 1.01
        self.G_Gain = wp_mean.max() / wp_mean[1] * 1
        self.B_Gain = wp_mean.max() / wp_mean[2] * 1.13
        print("R_Gain:", self.R_Gain)
        print("G_Gain:", self.G_Gain)
        print("B_Gain:", self.B_Gain)
        self.white_point = np.array([1/self.R_Gain, 1/self.G_Gain, 1/self.B_Gain])

        return

    def multilight_wp_wb_button_function(self):
        self.raw_image = self.get_image()
        if self.image_format.lower() in ["raw"]:
            self.raw_image = self.raw_image_isp(self.raw_image)
        else:
            self.raw_image = self.raw_image / 255.

        self.multilight_wb_gain = self.raw_image[..., 1:2] / self.raw_image
        self.multilight_wb_gain[..., 0] *= 1.01
        self.multilight_wb_gain[..., 2] *= 1.13
        white_point = 1 / self.multilight_wb_gain

        white_point_XYZ = smv_colour.RGB2XYZ(torch.from_numpy(np.float32(white_point)), "bt709")
        white_point_xyY = smv_colour.XYZ2xyY(white_point_XYZ)
        x, y = white_point_xyY[..., 0], white_point_xyY[..., 1]
        n = (x - 0.3320) / (y - 0.1858)
        cct = -449 * n ** 3 + 3525 * n ** 2 - 6823.3 * n + 5520.33
        cct = np.array(cct)

        # np.save("cct.npy", cct)

        # data = np.load("calibration_3nh.npy", allow_pickle=True).item()
        cct1 = 4739.57958984375
        cct2 = 6502.56591796875
        cct[cct > cct2] = cct2
        cct[cct < cct1] = cct1


        alpha = (1 / cct - 1 / cct2) / (1 / cct1 - 1 / cct2)
        self.multilight_alpha = alpha[..., None]
        return

    def show_image_button_function(self):
        th = threading.Thread(target=self.Display)
        th.start()

    def show_image_in_Qlabel(self, image, viewer):
        img3 = QtGui.QImage(image, image.shape[1], image.shape[0], image.shape[1] * 3, QtGui.QImage.Format_RGB888)
        # image2 = QtGui.QPixmap(img3).scaled(viewer.width(), viewer.height())
        image2 = QtGui.QPixmap(img3)

        viewer.setPixmap(image2)
        # viewer.setScaledContents(True)

    def cc_wb_button_function(self):
        try:
            self.raw_image = self.get_image()

            if self.image_format.lower() in ["raw"]:
                self.raw_image = self.raw_image_isp(self.raw_image)
            else:
                self.raw_image = self.raw_image / 255.
            _, charts_rgb, _ = mcc_detect_color_checker.detect_color_checker(self.raw_image)

            white_block = charts_rgb[18]
            self.R_Gain = white_block.max() / white_block[0]
            self.G_Gain = white_block.max() / white_block[1]
            self.B_Gain = white_block.max() / white_block[2]
            self.white_point = np.array([1 / self.R_Gain, 1 / self.G_Gain, 1 / self.B_Gain])

            print("R_Gain:", self.R_Gain)
            print("G_Gain:", self.G_Gain)
            print("B_Gain:", self.B_Gain)


            print("cc_wb_button_function finished!!!")
            # self.show_image_in_Qlabel(marker_image, self.ui.result_viewer)
        except Exception as e:
            print(e)
        return

    def calibration_button_function(self):
        self.raw_image = self.get_image()
        if self.calib_image_num > 1:
            self.raw_image = np.int32(self.raw_image)
            for i in range(self.calib_image_num-1):
                self.raw_image = self.raw_image + np.int32(self.get_image())
        self.raw_image = self.raw_image / self.calib_image_num


        if self.image_format.lower() in ["raw"]:
            self.raw_image = self.raw_image_isp(self.raw_image)
        else:
            print(self.raw_image.shape)
            self.raw_image = self.raw_image[..., ::-1] / 255.
            self.raw_image = np.float32(self.raw_image)



        sorted_centroid, charts_RGB, marker_image = mcc_detect_color_checker.detect_color_checker(self.raw_image)
        cv2.imwrite("marker_image.jpg", marker_image)
        color_calibration = ColorCalibration()
        wb_gain = color_calibration.calculate_WBgain(charts_RGB)

        Y_gain = color_calibration.calculate_Ygain(charts_RGB * wb_gain[None])
        wb_charts_RGB = (charts_RGB * wb_gain[None] * Y_gain)[:, None]
        ccm = color_calibration.calculate_ccm(np.float64(wb_charts_RGB), 1, self.ideal_lab)
        image_calib = color_correction.image_correction(self.raw_image, "linear", wb_gain, Y_gain, ccm, True,
                                                        True, True)

        deltaC, deltaE = color_correction.evaluate_result(image_calib, "linear", sorted_centroid, self.ideal_lab)
        image_gt = color_correction.draw_gt_in_image(image_calib, "linear", deltaC, sorted_centroid, self.ideal_linear_rgb)

        self.show_image_in_Qlabel(np.uint8(image_gt * 255), self.ui.result_viewer)
        cct = float(color_science.calculate_cct(1 / wb_gain))
        cv2.imwrite("image_gt.png", np.uint8(image_gt[..., ::-1] ** (1 / 2.2) * 255))



        if os.path.exists(self.result_path):
            cct_ccm_list = np.load(self.result_path, allow_pickle=True).item()
            # os.remove(result_path)
            cct_ccm_list[cct] = ccm
            np.save(self.result_path, cct_ccm_list)
        else:
            np.save(self.result_path, {cct: ccm})

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
        self.init_camera(method, value_int)
        self.stopEvent.clear()
        self.show_image_button_function()

    def quick_calibration_button_function(self):
        self.ser.write("<1,S=ON>".encode("gbk"))
        for color_temp in range(0, 101, 10):
            time.sleep(1)
            self.ser.write(("<1,C=%d>"%color_temp).encode("gbk"))
            time.sleep(3)
            self.calibration_button_function()

        self.ser.write("<1,S=OFF>".encode("gbk"))
        return

    def save_image_button_function(self):
        self.raw_image = self.get_image()
        if self.calib_image_num > 1:
            self.raw_image = np.int32(self.raw_image)
            for i in range(self.calib_image_num-1):
                self.raw_image = self.raw_image + np.int32(self.get_image())
        self.raw_image = self.raw_image / self.calib_image_num
        if self.image_format.lower() in ["raw"]:
            self.raw_image = self.raw_image_isp(self.raw_image)
        cv2.imwrite("save_image_%f.png"%time.time(), np.uint16(self.raw_image*65535))
        return

    def Display(self):
        while True:
            frame = self.get_image()
            if self.image_format.lower() in ["png", "jpg", "jpeg", "tiff", "bmp"]:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            if self.image_format.lower() == "raw":
                img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_Grayscale16)

            pix_img = QPixmap.fromImage(img)
            # pix_img = pix_img.scaled(frame.shape[1]//8, frame.shape[0]//8, QtCore.Qt.KeepAspectRatio)
            self.ui.raw_image_viewer.setPixmap(pix_img)

            cv2.waitKey(1)
            # 判断关闭事件是否已触发
            if True == self.stopEvent.is_set():
                # 关闭事件置为未触发，清空显示label
                self.stopEvent.clear()
                self.ui.raw_image_viewer.clear()
                break
'''