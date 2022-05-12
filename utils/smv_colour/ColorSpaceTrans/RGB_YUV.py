import numpy as np

class YUVTransfer:
    def __init__(self, color_space):

        assert color_space.lower() in ['sdtv_bt470', 'hdtv_bt709'], 'please select color_space in sdtv_bt470,hdtv_bt709'

        if color_space == 'sdtv_bt470':
            self.W_R, self.W_G, self.W_B = 0.2990, 0.5870, 0.1140
        elif color_space == 'hdtv_bt709':
            self.W_R, self.W_G, self.W_B = 0.2126, 0.7152, 0.0722
        self.U_max = 0.436
        self.V_max = 0.615

    def rgb2yuv(self, inp):
        """
        :param inp:  the shape of input should be [h, w, c], and the data range should be [0, 1]
        """
        img_type = inp.dtype
        r, g, b = inp[:, :, 0], inp[:, :, 1], inp[:, :, 2]

        y = self.W_R * r + self.W_G * g + self.W_B * b
        u = -self.U_max * self.W_R / (1 - self.W_B) * r - self.U_max * self.W_G / (1 - self.W_B) * g + self.U_max * b
        v = self.V_max * r - self.V_max * self.W_G / (1 - self.W_R) * g - self.V_max * self.W_B / (1 - self.W_R) * b

        # print(self.W_R, self.W_G, self.W_B)
        # print(-self.U_max * self.W_R / (1 - self.W_B), -self.U_max * self.W_G / (1 - self.W_B), self.U_max)
        # print(self.V_max, -self.V_max * self.W_G / (1 - self.W_R), -self.V_max * self.W_B / (1 - self.W_R))

        return np.concatenate((y[..., None].clip(0, 1), u[..., None].clip(-0.5, 0.5), v[..., None].clip(-0.5, 0.5)),
                              axis=-1).astype(img_type)

    def yuv2rgb(self, inp):
        """
        :param inp:  the shape of input should be [h, w, c], and the data range should be [0, 1]
        """
        img_type = inp.dtype
        y, u, v = inp[:, :, 0], inp[:, :, 1], inp[:, :, 2]

        r = y + (1 - self.W_R) / self.V_max * v
        g = y - self.W_B * (1 - self.W_B) / (self.U_max * self.W_G) * u - self.W_R * (1 - self.W_R) / (
                self.V_max * self.W_G) * v
        b = y + (1 - self.W_B) / self.U_max * u

        # print(1, 0, (1 - self.W_R) / self.V_max)
        # print(1, - self.W_B * (1 - self.W_B) / (self.U_max * self.W_G), - self.W_R * (1 - self.W_R) / (
        #         self.V_max * self.W_G))
        # print(1, (1 - self.W_B) / self.U_max, 0)

        return np.concatenate((r[..., None], g[..., None], b[..., None]), axis=-1).astype(img_type)

if __name__ == '__main__':
    yuv_trans = YUVTransfer('sdtv_bt470')
    a = np.zeros((1080, 1920, 3))
    re = yuv_trans.yuv2rgb(a)