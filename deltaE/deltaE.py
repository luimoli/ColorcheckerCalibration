import torch

class DeltaE_cmp:
    def __init__(self) -> None:
        self.RGB709_to_XYZ_matrix = torch.tensor([[0.412391, 0.357584, 0.180481],
                                                  [0.212639, 0.715169, 0.072192],
                                                  [0.019331, 0.119195, 0.950532]], dtype=torch.float32)

        self.XYZ_to_RGB709_matrix = torch.tensor([[3.2409663, -1.5373788, -0.49861172],
                                                  [-0.96924204, 1.8759652, 0.04155577],
                                                  [0.05562956, -0.20397693, 1.0569717]], dtype=torch.float32)

        self.RGB2020_to_XYZ_matrix = torch.tensor([[0.636958, 0.144617, 0.168881],
                                                   [0.262700, 0.677998, 0.059302],
                                                   [0.000000, 0.028073, 1.060985]], dtype=torch.float32)

        self.XYZ_to_RGB2020_matrix = torch.tensor([[1.716651, -0.355671, -0.253366],
                                                   [-0.666684, 1.616481, 0.015769],
                                                   [0.017640, -0.042771, 0.942103]], dtype=torch.float32)

        self.cuda_available = torch.cuda.is_available()
        
        if self.cuda_available:
            self.RGB709_to_XYZ_matrix = self.RGB709_to_XYZ_matrix.cuda()
            self.RGB2020_to_XYZ_matrix = self.RGB2020_to_XYZ_matrix.cuda()
            self.XYZ_to_RGB2020_matrix = self.XYZ_to_RGB2020_matrix.cuda()
            self.XYZ_to_RGB709_matrix = self.XYZ_to_RGB709_matrix.cuda()

    def delta_E_CIE2000(self, Lab_1, Lab_2):
        L_1, a_1, b_1 = [Lab_1[..., x] for x in range(Lab_1.shape[-1])]
        L_2, a_2, b_2 = [Lab_2[..., x] for x in range(Lab_2.shape[-1])]

        k_L = 1
        k_C = 1
        k_H = 1

        l_bar_prime = 0.5 * (L_1 + L_2)

        c_1 = torch.hypot(a_1, b_1)
        c_2 = torch.hypot(a_2, b_2)

        c_bar = 0.5 * (c_1 + c_2)
        c_bar7 = c_bar ** 7
        g = 0.5 * (1 - torch.sqrt(c_bar7 / (c_bar7 + 25 ** 7)))
        a_1_prime = a_1 * (1 + g)
        a_2_prime = a_2 * (1 + g)
        c_1_prime = torch.hypot(a_1_prime, b_1)
        c_2_prime = torch.hypot(a_2_prime, b_2)
        c_bar_prime = 0.5 * (c_1_prime + c_2_prime)
        h_1_prime = torch.rad2deg(torch.atan2(b_1, a_1_prime)) % 360
        h_2_prime = torch.rad2deg(torch.atan2(b_2, a_2_prime)) % 360

        h_bar_prime = torch.where(
            torch.abs(h_1_prime - h_2_prime) <= 180,
            0.5 * (h_1_prime + h_2_prime),
            (0.5 * (h_1_prime + h_2_prime + 360)))

        t = (1 - 0.17 * torch.cos(torch.deg2rad(h_bar_prime - 30)) +
             0.24 * torch.cos(torch.deg2rad(2 * h_bar_prime)) +
             0.32 * torch.cos(torch.deg2rad(3 * h_bar_prime + 6)) -
             0.20 * torch.cos(torch.deg2rad(4 * h_bar_prime - 63)))

        h = h_2_prime - h_1_prime
        delta_h_prime = torch.where(h_2_prime <= h_1_prime, h - 360, h + 360)
        delta_h_prime = torch.where(torch.abs(h) <= 180, h, delta_h_prime)
        delta_L_prime = L_2 - L_1
        delta_C_prime = c_2_prime - c_1_prime
        delta_H_prime = (2 * torch.sqrt(c_1_prime * c_2_prime) * 
                            torch.sin(torch.deg2rad(0.5 * delta_h_prime)))

        s_L = 1 + ((0.015 * (l_bar_prime - 50) * (l_bar_prime - 50)) /
                   torch.sqrt(20 + (l_bar_prime - 50) * (l_bar_prime - 50)))
        s_C = 1 + 0.045 * c_bar_prime
        s_H = 1 + 0.015 * c_bar_prime * t

        delta_theta = (30 * torch.exp(-((h_bar_prime - 275) / 25) * ((h_bar_prime - 275) / 25)))
        c_bar_prime7 = c_bar_prime ** 7
        r_C = torch.sqrt(c_bar_prime7 / (c_bar_prime7 + 25 ** 7))
        r_T = -2 * r_C * torch.sin(torch.deg2rad(2 * delta_theta))

        d_E = torch.sqrt((delta_L_prime / (k_L * s_L)) ** 2 +
                         (delta_C_prime / (k_C * s_C)) ** 2 +
                         (delta_H_prime / (k_H * s_H)) ** 2 +
                         (delta_C_prime / (k_C * s_C)) * (delta_H_prime / (k_H * s_H)) * r_T)

        return d_E
