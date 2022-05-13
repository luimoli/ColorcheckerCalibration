import sys
sys.path.append("..")
import cv2.cv2 as cv2
from utils import mvsdk
import numpy as np
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007, demosaicing_CFA_Bayer_bilinear


def init_camera(exposure_method, exposure_time):
    # 枚举相机
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        print("No camera was found!")
        return

    for i, DevInfo in enumerate(DevList):
        print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
    i = 0 if nDev == 1 else int(input("Select camera: "))
    DevInfo = DevList[i]
    print(DevInfo)

    # 打开相机
    try:
        hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
    except mvsdk.CameraException as e:
        print("CameraInit Failed({}): {}".format(e.error_code, e.message))
        return

    # 获取相机特性描述
    cap = mvsdk.CameraGetCapability(hCamera)

    # 判断是黑白相机还是彩色相机
    monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

    # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
    if monoCamera:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
    else:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

    # 相机模式切换成连续采集
    mvsdk.CameraSetTriggerMode(hCamera, 0)

    mvsdk.CameraSetGain(hCamera, 100, 100, 100)

    # 手动曝光，曝光时间30ms  0:手动  1:自动
    mvsdk.CameraSetAeState(hCamera, exposure_method)
    if exposure_method == 0:
        mvsdk.CameraSetExposureTime(hCamera, exposure_time * 1000)
    else:
        mvsdk.CameraSetAeTarget(hCamera, exposure_time)

    # 让SDK内部取图线程开始工作
    mvsdk.CameraPlay(hCamera)
    return hCamera


def get_image(hCamera, image_format):
        # 从相机取一帧图片
    try:
        cap = mvsdk.CameraGetCapability(hCamera)
        FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * 3
        pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
        pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 2000)
        if image_format.lower() in ["png", "jpg", "jpeg", "tiff", "bmp"]:
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
            mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

            # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
            # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth,
                                   1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
        elif image_format.lower() in ["raw"]:
            # mvsdk.CameraFlipFrameBuffer(pRawData, FrameHead, 1)

            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pRawData)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

            # mvsdk.CameraSaveImage(self.hCamera, "./aaa_%.4d.raw", pRawData, FrameHead, mvsdk.FILE_RAW_16BIT, 100)

            frame_16bit = np.empty(int(frame.shape[0] / 1.5), dtype=np.uint16)
            frame_0, frame_1, frame_2 = frame[0::3], frame[1::3], frame[2::3]
            frame_16bit[0::2] = np.uint16(frame_0) * 256 + np.uint16(frame_1) % 16 * 16
            frame_16bit[1::2] = np.uint16(frame_2) * 256 + np.uint16(frame_1) // 16 * 16
            frame = frame_16bit.reshape((FrameHead.iHeight, FrameHead.iWidth))
        else:
            assert 0, "image_format should be in [png, jpg, jpeg, tiff, bmp, raw]"
        return frame
    except mvsdk.CameraException as e:
        print("===============================================")
        print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))
        return np.zeros((640, 480))


def inverse_tone_mapping(raw_image):
    raw_image = raw_image - 512
    raw_image = raw_image / 16
    raw_image = np.where(raw_image <= 30, 1.97130908 * raw_image - 29.56680552, raw_image) + 30
    result_image = raw_image / 4096
    return result_image


def raw_image_isp(raw_image):
    raw_image = inverse_tone_mapping(raw_image)
    raw_image = demosaicing_CFA_Bayer_bilinear(raw_image, pattern="RGGB")
    # cv2.imwrite("raw_image.png", raw_image*255)
    return np.float32(raw_image)


def get_rgb_image(hCamera, image_format, image_num=1):
    raw_image = get_image(hCamera, image_format)
    if image_num > 1:
        raw_image = np.int32(raw_image)
        for i in range(image_num - 1):
            raw_image = raw_image + np.int32(get_image(hCamera, image_format))
        raw_image = raw_image / image_num
    if image_format in ["raw"]:
        # np.save("raw_image2.npy", raw_image_isp(raw_image))
        return raw_image_isp(raw_image)[..., ::-1]
    else:
        return np.float32(raw_image[..., ::-1] / 255.)