import numpy as np
import torch
import colour

from .utils.func import split, stack

def lab2lch(Lab):
    """[Converts from *CIE Lab* colourspace to *CIE LCH* colourspace.]
    Args:
        Lab ([type]): [description]
    Returns:
        [type]: [description]
    """
    L, a, b = split(Lab)

    H = 180 * torch.atan2(b, a) / (torch.acos(torch.zeros(1)).item() * 2)
    H[H < 0] += 360
    C = torch.sqrt(a ** 2 + b ** 2)

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

    a_lab = C * torch.cos(torch.deg2rad(H))
    b_lab = C * torch.sin(torch.deg2rad(H))

    Lab = stack((L, a_lab, b_lab))

    return Lab

if __name__ == '__main__':
    randxyz = torch.rand((1080,1920,3), dtype=torch.float32)
    randlab = colour.XYZ_to_Lab(randxyz)
    randlab = torch.from_numpy(randlab)

    # # verify Lab_to_LCH----
    # cs = colour.Lab_to_LCHab(randlab)
    # cs = torch.from_numpy(cs)
    # our = Lab_to_LCH(randlab)
    # diff = cs - our
    # print(diff.max(), diff.mean())

    # # # verify LCH_to_Lab----
    # randlch = colour.Lab_to_LCHab(randlab)
    # randlch = torch.from_numpy(randlch)
    # cs = colour.LCHab_to_Lab(randlch)
    # cs = torch.from_numpy(cs)
    # our = lch2lab(randlch)
    # diff = abs(cs - our)
    # print(diff.max(), diff.mean())

