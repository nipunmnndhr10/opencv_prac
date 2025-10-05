import cv2
import random

img = cv2.imread('assets/823f.jpg', -1)
img = cv2.resize(img, (0,0), fx= 0.5, fy=0.5)
img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

# save image after manipulation
# cv2.imwrite('assets/my-notion-face-portrait.png', img)  # name of the file you want to store the img to, & source img

# opencv and numpy are closely related
# opencv turns its input image into a numpy array
# so if we
# print(img) # it gives us a numpy array
# print(type(img))
# print(img.shape) # provides the height (rows), width(cols) and channels of the img

# display image
# cv2.imshow('Image', img)  # window name, image
# cv2.waitKey(0)    # wait infinite amount of time, 5 means wait 5 seconds
# cv2.destroyAllWindows()

# -1, cv2.IMREADER_COLOR : Loads a color image. Any transparency of image will be neglected. It is default.
# 0, cv2.IMREAD_GRAYSCALE: loads image in grayscale mode
# 1, cv2.IMREAD_UNCHANGED : loads image as such including alpha channel
