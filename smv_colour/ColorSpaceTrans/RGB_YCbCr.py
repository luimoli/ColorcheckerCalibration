import numpy as np

class YcbcrTransfer:
    def __init__(self, color_space):
        """
                     BT.601    BT.709    BT.2020
        :param K_R:  0.2990    0.2126    0.2627
        :param K_G:  0.5870    0.7152    0.6780
        :param K_B:  0.1140    0.0722    0.0593
        """
        assert color_space.lower() in ['bt601', 'bt709', 'bt2020', 'bt2020nc', 'smpte', 'smpte_240m']

        if color_space == 'bt601':
            self.K_R, self.K_G, self.K_B = 0.2990, 0.5870, 0.1140
        elif color_space == 'bt709':
            self.K_R, self.K_G, self.K_B = 0.2126, 0.7152, 0.0722
        elif color_space == 'bt2020' or color_space == 'bt2020nc':
            self.K_R, self.K_G, self.K_B = 0.2627, 0.6780, 0.0593
        elif color_space == 'smpte' or color_space == 'smpte_240m':
            # self.K_R, self.K_G, self.K_B = 0.2120, 0.7010, 0.0870   # from wiki
            self.K_R, self.K_G, self.K_B = 0.2122, 0.7013, 0.0865  # from colour_science

    def rgb2ycbcr(self, inp, limit2full=True):
        """
        :param inp:  the shape of input should be [h, w, c], and the data range should be [0, 1]
        :param limit2full: convert limited range to full range
        transfer_matrix: [[        K_R,                 K_G,                 K_B         ],
                          [-0.5*(K_R/(1-K_B)),  -0.5*(K_G/(1-K_B)),          0.5         ],
                          [        0.5,         -0.5*(K_G/(1-K_R)),  -0.5*(K_B/(1 - K_R))]]
        """
        img_type = inp.dtype
        r, g, b = inp[:, :, 0], inp[:, :, 1], inp[:, :, 2]
        y = self.K_R * r + self.K_G * g + self.K_B * b
        pb = -0.5 * (self.K_R / (1 - self.K_B)) * r - 0.5 * (self.K_G / (1 - self.K_B)) * g + 0.500000 * b
        pr = 0.500000 * r - 0.5 * (self.K_G / (1 - self.K_R)) * g - 0.5 * (self.K_B / (1 - self.K_R)) * b
        if limit2full:
            y = (219 / 255.) * y + 16 / 255.
            cb = (224 / 255.) * pb
            cr = (224 / 255.) * pr
            return np.concatenate(
                (y[..., None].clip(0, 1), cb[..., None].clip(-0.5, 0.5), cr[..., None].clip(-0.5, 0.5)),
                axis=-1).astype(img_type)

        else:
            return np.concatenate(
                (y[..., None].clip(0, 1), pb[..., None].clip(-0.5, 0.5), pr[..., None].clip(-0.5, 0.5)),
                axis=-1).astype(img_type)

    def ycbcr2rgb(self, inp, limit2full=True):
        """
        :param inp:  the shape of input should be [h, w, c], and the data range should be [0, 1]
        :param limit2full: convert limited range to full range
        transfer_matrix: [[1,        0,                  2-2*K_R     ],
                          [1, -K_B*(2-2*K_B)/K_G,  -K_R*(2-2*K_R)/K_G],
                          [1,    (2-2*K_B),                0         ]]
        """
        img_type = inp.dtype
        y, cb, cr = inp[:, :, 0], inp[:, :, 1], inp[:, :, 2]
        if limit2full:
            y = (y - 16 / 255.) * (255 / 219.)
            cb = cb * (255 / 224.)
            cr = cr * (255 / 224.)
        r = y + (2 - 2 * self.K_R) * cr
        g = y - self.K_B * (2 - 2 * self.K_B) / self.K_G * cb - self.K_R * (2 - 2 * self.K_R) / self.K_G * cr
        b = y + (2 - 2 * self.K_B) * cb
        return np.concatenate((r[..., None], g[..., None], b[..., None]), axis=-1).astype(img_type)