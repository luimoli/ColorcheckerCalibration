#================================
# use imatest to read checker
#================================



import numpy as np
import cv2
import torch

if __name__ == '__main__':
    image_path = r"./data/1.png"
    # image_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) / 255.
    image_bgr = cv2.imread(image_path) / 255.
    image_rgb = np.float32(image_bgr[..., ::-1].copy())
    img_cut = image_rgb[560:2432, 526:3356, :].copy()
    
    img_cut = img_cut[::4, ::4, :].copy()
    # cv2.imwrite(r"./data/1_cut.png", (img_cut[..., ::-1] * 255.))

    img_cut = img_cut * 255.

    image_rgb = torch.from_numpy(image_rgb)
    # result_rgb_image = checker.conversion(image_rgb)
    # result_rgb_image = result_rgb_image.cpu().numpy()
    # cv2.imwrite(r"D:\\Code\\VideoHDR-mm\\example-data\\0000002_sdr_no.png", np.uint8(result_rgb_image[..., ::-1] * 255.))
    # cv2.imwrite(r"./img/A_res.JPG", (result_rgb_image[..., ::-1] * 255.))