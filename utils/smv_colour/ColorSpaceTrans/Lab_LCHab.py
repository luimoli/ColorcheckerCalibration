import numpy as np

from .utils.func import split, stack

def lab2lch(Lab):
    """[Converts from *CIE Lab* colourspace to *CIE LCH* colourspace.]
    Args:
        Lab ([type]): [description]
    Returns:
        [type]: [description]
    """
    L, a, b = split(Lab)

    H = 180 * np.arctan2(b, a) / (np.arccos(np.zeros(1)).item() * 2)
    H[H < 0] += 360
    C = np.sqrt(a ** 2 + b ** 2)

    LCH = stack((L, C, H))

    return LCH

def lch2lab(LCH):
    """[Converts from *CIE LCH* colourspace to *CIE Lab* colourspace.]
    Args:
        LCHab ([type]): [description]
    Returns:
        [type]: [description]
    """
    L, C, H = split(LCH)

    a_lab = C * np.cos(np.deg2rad(H))
    b_lab = C * np.sin(np.deg2rad(H))

    Lab = stack((L, a_lab, b_lab))

    return Lab

if __name__ == '__main__':
    import colour
    randxyz = np.float32(np.random.random((1080,1920,3)))
    randlab = colour.XYZ_to_Lab(randxyz)

    
    # # verify Lab_to_LCH----
    # cs = colour.Lab_to_LCHab(randlab)
    # our = lab2lch(randlab)
    # print(cs.max(),cs.mean())
    # diff = cs - our
    # print(diff.max(), diff.mean())

    # # # verify LCH_to_Lab----
    # randlch = colour.Lab_to_LCHab(randlab)
    # cs = colour.LCHab_to_Lab(randlch)
    # our = lch2lab(randlch)
    # diff = abs(cs - our)
    # print(diff.max(), diff.mean())

