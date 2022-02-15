import numpy as np
from numpy import genfromtxt
from minimize_ccm import ccm_calculate


if __name__ == '__main__':
    # # calculate ccm
    # lab_ideal = np.float32(genfromtxt("./data/real_lab_xrite.csv", delimiter=',')) # from X-rite
    lab_ideal = np.float32(genfromtxt("./data/real_lab.csv", delimiter=',')) # from imatest
    rgb_data = np.float32(genfromtxt("./data/measure_rgb_ck2.csv", delimiter=','))
    ccm_matrix = ccm_calculate(rgb_data, lab_ideal)