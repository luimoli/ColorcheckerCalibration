from .ColorSpaceTrans import RGB_YCbCr, RGB_YUV, XYZ_xyY, RGB_HSV, XYZ_Lab, XYZ_Luv, xy_CCT, Lab_LCHab, Luv_LCHuv, RGB_XYZ


def RGB2Ycbcr(img, color_space='bt709'):
    trans = RGB_YCbCr.YcbcrTransfer(color_space)
    return trans.rgb2ycbcr(img)
def Ycbcr2RGB(img, color_space='bt709'):
    trans = RGB_YCbCr.YcbcrTransfer(color_space)
    return trans.ycbcr2rgb(img)


def YUV2RGB(img, color_space='hdtv_bt709'):
    trans = RGB_YUV.YUVTransfer(color_space)
    return trans.yuv2rgb(img)
def RGB2YUV(img, color_space='hdtv_bt709'):
    trans = RGB_YUV.YUVTransfer(color_space)
    return trans.rgb2yuv(img)

def RGB2XYZ(img_rgb, color_space):
    """
        color_space ([str]): ['bt709', 'bt2020']
    """
    trans = RGB_XYZ.RGBXYZTransfer(color_space)
    return trans.rgb2xyz(img_rgb)
def XYZ2RGB(img_xyz, color_space):
    """
        color_space ([str]): ['bt709', 'bt2020']
    """
    trans = RGB_XYZ.RGBXYZTransfer(color_space)
    return trans.xyz2rgb(img_xyz)

def HSV2RGB(img_hsv):
    trans = RGB_HSV.HSVTransfer()
    return trans.hsv2rgb(img_hsv)
def RGB2HSV(img_rgb):
    trans = RGB_HSV.HSVTransfer()
    return trans.rgb2hsv(img_rgb)


def XYZ2xyY(img_XYZ):
    return XYZ_xyY.xyz2xyy(img_XYZ)
def xyY2XYZ(img_xyY):
    return XYZ_xyY.xyy2xyz(img_xyY)


def XYZ2Lab(img_xyz):
    return XYZ_Lab.xyz2lab(img_xyz)
def Lab2XYZ(img_lab):
    return XYZ_Lab.lab2xyz(img_lab)


def XYZ2Luv(img_xyz):
    return XYZ_Luv.xyz2luv(img_xyz)
def Luv2XYZ(img_luv):
    return XYZ_Luv.luv2xyz(img_luv)


def Lab2LCHab(img_lab):
    return Lab_LCHab.lab2lch(img_lab)
def LCHab2Lab(img_lch):
    return Lab_LCHab.lch2lab(img_lch)


def Luv2LCHuv(img_luv):
    return Luv_LCHuv.luv2lch(img_luv)
def LCHuv2Luv(img_lch):
    return Luv_LCHuv.lch2luv(img_lch)


def xy2xyY(xy):
    return XYZ_xyY.xy2xyy(xy)

def uv2xy(uv):
    return XYZ_Luv.uv2xy(uv)


def xy2CCT(xy):
    return xy_CCT.xy2CCT_mccamy1992(xy)

