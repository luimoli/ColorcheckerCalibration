from colour import colour
import numpy as np
import cv2

# from utils import smv_colour

# res = smv_colour.XYZ2Lab(np.array([1,0,1]))
# print(res)



# img = cv2.imread('./2022-05-26_17-10-18_408.png')
# cv2.imwrite('./2022-05-26_17-10-18_408_gamma.png', ((img/255.) ** (1/2.2)) * 255.)


img = cv2.imread(r'data\tmp\626b24cc-5994-4ff0-bc83-419143ea9b9d.jpg')
cv2.imwrite(r'data\tmp\626b24cc-5994-4ff0-bc83-419143ea9b9d_linear.jpg', ((img/255.) ** (2.2)) * 255.)

