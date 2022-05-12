import numpy as np

from .utils.func import split, stack
from .utils.constants import const
from .XYZ_xyY import xyy2xyz, xy2xyy


def xyz2luv(XYZ, illuminant=const.ILLUMINANTS['D65']):
    """[Converts from *CIE XYZ* tristimulus values to *CIE Luv* colourspace.]
    Args:
        XYZ ([arr][0,1]): [*CIE XYZ* tristimulus values.]
        illuminant ([arr], optional): [Reference *illuminant* *xy* chromaticity coordinates]. Defaults to CIE D65.
    Returns:
        [arr]: [*CIE Luv* colourspace array.]
    """

    X, Y, Z = split(XYZ)
    X_r, Y_r, Z_r = split(xyy2xyz(xy2xyy(illuminant)))
    
    y_r = Y / Y_r
    L = np.where(y_r > const.CIE_E, 116 * y_r ** (1 / 3) - 16, const.CIE_K * y_r)

    X_Y_Z = X + 15 * Y + 3 * Z
    X_r_Y_r_Z_r = X_r + 15 * Y_r + 3 * Z_r
    u = (13 * L * ((4 * X / X_Y_Z) - (4 * X_r / X_r_Y_r_Z_r)))
    v = (13 * L * ((9 * Y / X_Y_Z) - (9 * Y_r / X_r_Y_r_Z_r)))

    Luv = stack((L, u, v))

    return Luv

def luv2xyz(Luv, illuminant=const.ILLUMINANTS['D65']):
    """[Converts from *CIE Luv* colourspace to *CIE XYZ* tristimulus values.]
    Args:
        Luv ([arr]): [*CIE Luv* colourspace array.]
        illuminant ([arr], optional): [Reference *illuminant* *xy* chromaticity coordinates]. Defaults to CIE D65.
    Returns:
        [arr][0,1]: [*CIE XYZ* tristimulus values.]
    """

    L, u, v = split(Luv)

    X_r, Y_r, Z_r = split(xyy2xyz(xy2xyy(illuminant)))

    Y = np.where(L > const.CIE_E * const.CIE_K, ((L + 16) / 116) ** 3, L / const.CIE_K)

    a = 1 / 3 * ((52 * L / (u + 13 * L * (4 * X_r / (X_r + 15 * Y_r + 3 * Z_r)))) - 1)
    b = -5 * Y
    c = -1 / 3.0
    d = Y * (39 * L / (v + 13 * L * (9 * Y_r / (X_r + 15 * Y_r + 3 * Z_r))) - 5)

    X = (d - b) / (a - c)
    Z = X * a + b

    XYZ = stack((X, Y, Z))

    return XYZ

def uv2xy(uv):
    """[Returns the *xy* chromaticity coordinates from given *CIE Luv* colourspace]
    Args:
        uv ([arr][0,1]): [*CIE Luv u"v"* chromaticity coordinates.]
    Returns:
        [arr][0,1]: [*xy* chromaticity coordinates.]
    """

    u, v = split(uv)
    xy = stack((9 * u / (6 * u - 16 * v + 12), 4 * v / (6 * u - 16 * v + 12)))
    return xy


if __name__ =='__main__':
    randxyz = np.rand((1080,1920,3), dtype=np.float32)

    # our = xyz2luv(randxyz)
    # cs = colour.XYZ_to_Luv(randxyz)
    # cs = np.from_numpy(cs)
    # diff = abs(cs - our)
    # print(diff.max(), diff.mean())

    # randxyz = randxyz.numpy()
    # randluv = colour.XYZ_to_Luv(randxyz)
    # randluv = np.from_numpy(randluv)
    # cs = colour.Luv_to_XYZ(randluv)
    # cs = np.from_numpy(cs)
    # our = luv2xyz(randluv)
    # diff = abs(cs - our)
    # print(diff.max(), diff.mean())

    
