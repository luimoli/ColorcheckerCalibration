import cv2.cv2 as cv2

if __name__ == '__main__':
    image = cv2.imread("../data/mindvision/mv_2300.PNG")[..., ::-1] / 255.
    cct_ccm_dict = np.load(".npy", allow_pickle=True).item
    image_color_correction = ImageColorCorrection(cct_ccm_dict, "linear", "cc")
    image_color_correction.doWhiteBalance(wb_image=image)
    image_color_correction.correctImage(image, "linear")