import cv2

from PIL import Image
from PIL import ImageEnhance

import toolkit.datasets.pre_process as pp


img = cv2.imread("I:\\tracker\\auto_track_final\\OTB100\\Singer1\\img\\0095.jpg")
cv2.imshow('1', img)
cv2.waitKey(30)
img_process = pp.im_enhance_1(img)
cv2.imshow('2', img_process)
cv2.waitKey(0)
# img = Image.open("I:\\tracker\\auto_track_final\\OTB100\\Singer1\\img\\0095.jpg")
# enh_con = ImageEnhance.Contrast(img)
# contrast = 1.5
# img_process = enh_con.enhance(contrast)
# img_process.show()