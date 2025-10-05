import numpy as np 
import cv2

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

# Isolating specific part of the image

while True:
   ret, frame = cap.read()

   width = int(cap.get(3))
   height = int(cap.get(4))

   # convert bgr image into hsv
   hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

   # extracting required color from the 'frame': set lower and upper bound of the pixels (colors) we want to be displayed
   lower_blue = np.array([90,50,50])  # light blue hsv
   upper_blue = np.array([130, 255, 255]) # darker blue hsv

   # create mask: is a portion of an image/frame
   mask = cv2.inRange(hsv, lower_blue, upper_blue) # only all of the blue pixels in the range will be displayed


   # use the mask: apply it to the image (frame)
   result = cv2.bitwise_and(frame, frame, mask = mask)


   cv2.imshow('Isolated Color', result)
   cv2.imshow('Mask of Isolated Color (in Binary)', mask)

   if cv2.waitKey(1) == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()

# converting bgr image into hsv color

# BGR_color = np.array([[[255,0,0]]], dtype = np.uint8) # one pixel
# # cv2.cvtColor expects an image (a pixel) that has a specific shape (rows, cols, channels) not a color
# hsv_color = cv2.cvtColor(BGR_color, cv2.COLOR_BGR2HSV)
# print("HSV color:",hsv_color)