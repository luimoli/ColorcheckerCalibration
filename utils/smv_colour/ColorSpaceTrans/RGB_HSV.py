import cv2
import numpy as np
class HSVTransfer:
    def rgb2hsv(self, img_arr):
        r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
        maximum = np.amax(img_arr, axis=-1)
        minimum = np.amin(img_arr, axis=-1)
        v = maximum
        s = np.zeros_like(v)
        s = np.where(maximum != 0., (maximum - minimum) / maximum, s)

        h = np.zeros_like(v)
        delta = maximum - minimum
        h = np.where((maximum == r) & (delta != 0), (60.0 * ((g - b) / delta) + 360) % 360.0, h)
        h = np.where((maximum == g) & (delta != 0), 60.0 * ((b - r) / delta) + 120.0, h)
        h = np.where((maximum == b) & (delta != 0), 60.0 * ((r - g) / delta) + 240.0, h)

        return np.concatenate((h[..., None], s[..., None], v[..., None]), axis=-1)

    def hsv2rgb(self, img_arr):
        h, s, v = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]

        h_floored = np.floor(h)
        h_sub_i = np.div(h_floored, 60, rounding_mode='floor') % 6

        var_f = (h / 60.0) - np.div(h_floored, 60, rounding_mode='floor')
        var_p = v * (1.0 - s)
        var_q = v * (1.0 - var_f * s)
        var_t = v * (1.0 - (1.0 - var_f) * s)

        r, g, b = np.empty_like(h), np.empty_like(h), np.empty_like(h)

        r = np.where((h_sub_i == 0) | (h_sub_i == 5), v, r)
        r = np.where((h_sub_i == 1), var_q, r)
        r = np.where((h_sub_i == 2) | (h_sub_i == 3), var_p, r)
        r = np.where((h_sub_i == 4), var_t, r)

        g = np.where((h_sub_i == 0), var_t, g)
        g = np.where((h_sub_i == 1) | (h_sub_i == 2), v, g)
        g = np.where((h_sub_i == 3), var_q, g)
        g = np.where((h_sub_i == 4) | (h_sub_i == 5), var_p, g)

        b = np.where((h_sub_i == 0) | (h_sub_i == 1), var_p, b)
        b = np.where((h_sub_i == 2), var_t, b)
        b = np.where((h_sub_i == 3) | (h_sub_i == 4), v, b)
        b = np.where((h_sub_i == 5), var_q, b)

        return np.concatenate((r[..., None], g[..., None], b[..., None]), axis=-1)


if __name__ == '__main__':
    filename = '../img-s-uint8/cat.jpg'
    img_rgb = np.float32(cv2.imread(filename)[:, :, ::-1] / 255.)
    hsv_transfer = HSVTransfer()

    # # rgb to hsv align
    cv_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    our_hsv = hsv_transfer.rgb2hsv(img_rgb)
    our_hsv = our_hsv.cpu().numpy()
    diff = abs(cv_hsv - our_hsv)

    # # hsv to rgb align
    # cv_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    # cv_rgb = cv2.cvtColor(cv_hsv, cv2.COLOR_HSV2RGB)
    # our_rgb = hsv_transfer.hsv2rgb(np.from_numpy(cv_hsv))
    # our_rgb = our_rgb.cpu().numpy()
    # diff = abs(cv_rgb - our_rgb)


    diff1, diff2, diff3 = diff[:, :, 0], diff[:, :, 1], diff[:, :, 2]
    print(np.max(diff1), np.mean(diff1))
    print(np.max(diff2), np.mean(diff2))
    print(np.max(diff3), np.mean(diff3))
    print()
