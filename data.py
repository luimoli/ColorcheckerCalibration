import numpy as np
from numpy import genfromtxt

class Const(object):
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("Can't change const.%s" % name)
        if not name.isupper():
            raise self.ConstCaseError('const name "%s" is not all supercase' % name)

        self.__dict__[name] = value

const = Const()
const.ILLUMINANTS = {'D50': [ 0.3457,  0.3585], 
                    'D55': [ 0.33243, 0.34744], 
                    'D60': [0.32162624, 0.337737], 
                    'D65': [ 0.3127,  0.329]}
const.CIE_E = 0.008856451679035631
const.CIE_K = 903.2962962962963
const.K_M = 683
const.KP_M = 1700

const.RGB_XYZ = np.array([[0.412391, 0.357584, 0.180481],
                                  [0.212639, 0.715169, 0.072192],
                                  [0.019331, 0.119195, 0.950532]], dtype=np.float32)
const.REALRGB = np.float32(genfromtxt("./data/real_rgb.csv", delimiter=','))
const.NONLINEARRGB = np.float32(genfromtxt("./data/measure_rgb_d65.csv", delimiter=','))