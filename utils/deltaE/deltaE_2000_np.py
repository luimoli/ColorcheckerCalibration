import numpy as np

def delta_E_CIE2000(Lab_1, Lab_2, textiles=False):
    L_1, a_1, b_1 = [Lab_1[..., x] for x in range(Lab_1.shape[-1])]
    L_2, a_2, b_2 = [Lab_2[..., x] for x in range(Lab_2.shape[-1])]

    k_L = 2 if textiles else 1
    k_C = 1
    k_H = 1

    l_bar_prime = 0.5 * (L_1 + L_2)

    c_1 = np.hypot(a_1, b_1)
    c_2 = np.hypot(a_2, b_2)

    c_bar = 0.5 * (c_1 + c_2)
    c_bar7 = c_bar ** 7

    g = 0.5 * (1 - np.sqrt(c_bar7 / (c_bar7 + 25 ** 7)))

    a_1_prime = a_1 * (1 + g)
    a_2_prime = a_2 * (1 + g)
    c_1_prime = np.hypot(a_1_prime, b_1)
    c_2_prime = np.hypot(a_2_prime, b_2)
    c_bar_prime = 0.5 * (c_1_prime + c_2_prime)

    h_1_prime = np.degrees(np.arctan2(b_1, a_1_prime)) % 360
    h_2_prime = np.degrees(np.arctan2(b_2, a_2_prime)) % 360

    h_bar_prime = np.where(
        np.fabs(h_1_prime - h_2_prime) <= 180,
        0.5 * (h_1_prime + h_2_prime),
        (0.5 * (h_1_prime + h_2_prime + 360)),
    )

    t = (1 - 0.17 * np.cos(np.deg2rad(h_bar_prime - 30)) +
        0.24 * np.cos(np.deg2rad(2 * h_bar_prime)) +
        0.32 * np.cos(np.deg2rad(3 * h_bar_prime + 6)) -
        0.20 * np.cos(np.deg2rad(4 * h_bar_prime - 63)))

    h = h_2_prime - h_1_prime
    delta_h_prime = np.where(h_2_prime <= h_1_prime, h - 360, h + 360)
    delta_h_prime = np.where(np.fabs(h) <= 180, h, delta_h_prime)

    delta_L_prime = L_2 - L_1
    delta_C_prime = c_2_prime - c_1_prime
    delta_H_prime = (2 * np.sqrt(c_1_prime * c_2_prime) * np.sin(
        np.deg2rad(0.5 * delta_h_prime)))

    s_L = 1 + ((0.015 * (l_bar_prime - 50) * (l_bar_prime - 50)) /
            np.sqrt(20 + (l_bar_prime - 50) * (l_bar_prime - 50)))
    s_C = 1 + 0.045 * c_bar_prime
    s_H = 1 + 0.015 * c_bar_prime * t

    delta_theta = (
        30 * np.exp(-((h_bar_prime - 275) / 25) * ((h_bar_prime - 275) / 25)))

    c_bar_prime7 = c_bar_prime ** 7

    r_C = np.sqrt(c_bar_prime7 / (c_bar_prime7 + 25 ** 7))
    r_T = -2 * r_C * np.sin(np.deg2rad(2 * delta_theta))

    d_E = np.sqrt((delta_L_prime / (k_L * s_L)) ** 2 +
                (delta_C_prime / (k_C * s_C)) ** 2 +
                (delta_H_prime / (k_H * s_H)) ** 2 +
                (delta_C_prime / (k_C * s_C)) * (delta_H_prime /
                                                (k_H * s_H)) * r_T)


    # d_E[12] *= 1.5
    # d_E[13] *= 2
    # d_E[14] *= 3
    # d_E[13: 15] *=3
    # print("***********************")
    # print(Lab_1[6], Lab_2[6],d_E[6])
    # print(Lab_1[11], Lab_2[11],d_E[11])
    # print(Lab_1[14], Lab_2[14],d_E[14])
    # print(Lab_1[18], Lab_2[18],d_E[18])



    return d_E