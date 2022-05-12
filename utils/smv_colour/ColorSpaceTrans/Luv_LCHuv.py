import numpy as np

from .utils.func import split, stack

def luv2lch(Luv):
    """
    Converts from *CIE Luv* colourspace to *CIE LCHuv* colourspace.
    Parameters
    Luv :*CIE Luv* colourspace array.
    Returns
    ndarray
        *CIE LCHuv* colourspace array. in domain [0, 100].
    """

    L, u, v = split(Luv)

    H = 180 * np.arctan2(v, u) / (np.arccos(np.zeros(1)).item() * 2)
    H[H < 0] += 360
    C = np.sqrt(u ** 2 + v ** 2)

    LCHuv = stack((L, C, H))

    return LCHuv

def lch2luv(LCHuv):
    """
    Converts from *CIE LCHuv* colourspace to *CIE Luv* colourspace.

    """
    L, C, H = split(LCHuv)

    u = C * np.cos(np.deg2rad(H))
    v = C * np.sin(np.deg2rad(H))

    Luv = stack((L, u, v))

    return Luv


if __name__ == '__main__':
    randxyz = np.rand((1080,1920,3), dtype=np.float32)
    # randluv = colour.XYZ_to_Luv(randxyz)

    # # verify luv2lch----
    # cs = colour.Luv_to_LCHuv(randluv)
    # our = luv2lch(randluv)
    # diff = abs(cs - our)
    # print(diff.max(), diff.mean())

    # # # verify lch2luv----
    # randlch = colour.Luv_to_LCHuv(randluv)
    # cs = colour.LCHuv_to_Luv(randlch)
    # our = lch2luv(randlch)
    # diff = abs(cs - our)
    # print(diff.max(), diff.mean())