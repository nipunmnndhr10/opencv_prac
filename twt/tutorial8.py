import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# pretrained model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

while True:
   ret, frame = cap.read()

   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   faces = face_cascade.detectMultiScale(gray, 1.3, 5)   
   # srcimg, scale factor (to shrink img by %), minNeighbors - how many many candidate rectangles is required to be overlapping on a specific area before determining the larger area as a 'face' -> accuracy

   for (x,y,w,h) in faces: # faces provides a rectangle
      cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 5)

      # grab the area of face to find eyes
      roi_gray = gray[y:y+w, x:x+w]
      roi_color = frame[y:y+h, x:x+w]

      # eye detection
      eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
      for (ex, ey, ew, eh) in eyes:
         cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 5)


   cv2.imshow("Frame", frame)

   if cv2.waitKey(1) == ord('q'):
      break


cap.release()
cv2.destroyAllWindows()