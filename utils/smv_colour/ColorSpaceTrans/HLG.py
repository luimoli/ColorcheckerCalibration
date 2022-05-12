import numpy as np

def hlg_inverse_ootf(rgb_screen, alpha=1000, beta=0):
    Yd = 0.2627 * (rgb_screen[:, :, 0:1]) + 0.6780 * (rgb_screen[:, :, 1:2]) + 0.0593 * (rgb_screen[:, :, 2:])
    gamma = 1.2
    rgb_scene = ((Yd - beta) / alpha) ** ((1 - gamma) / gamma) * ((rgb_screen - beta) / alpha)
    return rgb_scene


def hlg_oetf(rgb2100):
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073
    rgb_oetf = np.where(rgb2100 <= 1 / 12, ((3 * rgb2100) ** 0.5), (a * np.log(12 * rgb2100 - b) + c))
    return rgb_oetf


def hlg_inverse_eotf(data, alpha, beta):
    step1 = hlg_inverse_ootf(data, alpha, beta)
    return hlg_oetf(step1)


def eotf_HLG_BT2100(rgb2020, L_W=1000, L_B=0):
    step1 = oetf_inverse_ARIBSTDB67(rgb2020) / 12.
    return ootf_HLG_BT2100_1(step1, L_B, L_W)


def ootf_HLG_BT2100_1(x, L_B, L_W, gamma=1.2):
    R_S, G_S, B_S = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    alpha = L_W - L_B
    beta = L_B
    Y_S = 0.2627 * R_S + 0.6780 * G_S + 0.0593 * B_S

    R_D = alpha * R_S * np.abs(Y_S) ** (gamma - 1) + beta
    G_D = alpha * G_S * np.abs(Y_S) ** (gamma - 1) + beta
    B_D = alpha * B_S * np.abs(Y_S) ** (gamma - 1) + beta

    RGB_D = np.stack([R_D, G_D, B_D], dim=2)

    return RGB_D


def oetf_inverse_ARIBSTDB67(E_p):
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    E = np.where(E_p <= 1, (E_p / 0.5) ** 2, np.exp((E_p - c) / a) + b)
    return E
