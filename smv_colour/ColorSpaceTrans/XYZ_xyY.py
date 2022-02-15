import colour
import torch

from .utils.func import split, stack
from .utils.constants import const


def xyy2xyz(xyY):
    """
    [Converts between *CIE XYZ* tristimulus values and *CIE xyY* colourspace with reference *illuminant*.]
    Args:
        xyY ([array_like]): [*CIE xyY* colourspace array in domain [0, 1].]
    Returns:
        [array_like]: [*CIE XYZ* tristimulus values array in domain [0, 1].]
    """
    x, y, Y = split(xyY)
    XYZ = torch.where((y == 0)[..., None], stack((y, y, y)), stack((x * Y / y, Y, (1 - x - y) * Y / y)))

    return XYZ


def xyz2xyy(XYZ, illuminant=const.ILLUMINANTS['D65']):
    """
    [Converts from *CIE XYZ* tristimulus values to *CIE xyY* colourspace with reference *illuminant*.]
    Args:
        XYZ ([array_like]): [*CIE XYZ* tristimulus values in domain [0, 1].]
    Returns:
        [type]: [*CIE xyY* colourspace array in domain [0, 1].]
    """
    X, Y, Z = split(XYZ)
    XYZ_n = torch.zeros(XYZ.shape)
    XYZ_n[..., 0:2] = torch.tensor(illuminant)

    # replace the point which contains 0 in XYZ-format to avoid zero-divide.
    xyY = torch.where(torch.all(XYZ == 0, axis=-1)[..., None], XYZ_n, stack((X / (X + Y + Z), Y / (X + Y + Z), Y))) 

    return xyY

def xy2xyy(xy, Y=1.0):
    """[Converts from xy chromaticity coordinates to *CIE xyY* colourspace.]
    Args:
        xy ([array]): [xy chromaticity coordinates]
        Y (float, optional): [description]. Defaults to 1.0.
    Returns:
        [xyY]: [description]
    """
    xy = torch.tensor(xy, dtype=torch.float32)
    shape = xy.shape
    if shape[-1] == 3:
        return xy
    x, y = split(xy)
    xyY = stack((x, y, torch.full(x.shape, Y)))
    return xyY


if __name__ == "__main__":
    randarr = torch.rand((1080,1920,3))

    # our = xyz2xyy(randarr)
    # our = our.numpy()
    # cs = colour.XYZ_to_xyY(randarr)
    # diff = abs(our - cs)
    # print(diff.max(), diff.mean())

    randxyy = colour.XYZ_to_xyY(randarr)
    randxyy = torch.from_numpy(randxyy)
    our = xyy2xyz(randxyy)
    cs = colour.xyY_to_XYZ(randxyy)
    cs = torch.from_numpy(cs)
    diff = abs(cs - our)
    print(diff.max(), diff.mean())

    