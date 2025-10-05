# CAMERAS AND VIDEOCAPTURE

import numpy as np
import cv2

# take hold of the camera
# here 0 would access the first webcam (in case of multiple webcams number can be changed)  
# else if want to load a video file then VideoCapture('filename')
cap = cv2.VideoCapture(0)  

# returns frame: which is the numpy  array that represents our image
# ret tells if this capture works properly (capture won't work if the webcam is occupied by some other software)
   # so ret return True if the read was successful

while True:
   ret, frame = cap.read()   # frame is the image from the camera

   # VideoCapture have properties which can be accessed via diff indexes
   width = int(cap.get(3))
   height = int(cap.get(4))


   # to turn the frame into four separate images
   #1. creating a blank canvas to put the images
   image = np.zeros(frame.shape, np.uint8) # shape, type of arr

   #2. shrink and copy the frame 4 times
   # resize the frame to half of it's orginal size
   # (0, 0) tells OpenCV to use fx and fy scaling factors instead of explicit dimensions.
   # fx=0.5 and fy=0.5: shrink both width and height to half.
   smaller_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

   # fit the smaller_frames to the image canvas
   image[:height//2, :width//2] = cv2.rotate(smaller_frame, cv2.ROTATE_180) # top left
   image[height//2:, :width//2] =  smaller_frame # bottom left
   image[:height//2, width//2:] = cv2.rotate(smaller_frame, cv2.ROTATE_180) # top right
   image[height//2:, width//2:] = smaller_frame # bottom right


   cv2.imshow('frame', image)  

   if cv2.waitKey(1) == ord('q'):   # ordinal val of 'q' is integer ascii val of 'q'
      break
   # here waitKey(1) is being called repeatedly, in each loop it wait 1ms for input key

# release the camera resource
cap.release()
cv2.destroyAllWindows()

