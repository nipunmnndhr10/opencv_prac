import numpy as np
import cv2

# edge, corner detection


# img = cv2.imread('assets/chessboard.png')
cap = cv2.VideoCapture(0)

draw_lines = False

while True:
   ret, frame = cap.read()

   # grayscale conversion
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   # shi-tomasi corner detection algo
   corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)   # src img, num of corners. min quality, min euclidean distance betn corners

   # convert corners into int
   corners = np.intp(corners)
   # print(corners) # default are floating point vals

   for corner in corners:  # gives an 2d array
      # print(corner)
      x, y = corner.ravel()   # get one corner coordinates, .ravel() flattens an array
      cv2.circle(frame, (x,y), 10, (255,127,0), -1)  # drawing circles in the corner coordinates


   # draw random colored lines between every corners

   if draw_lines:
      for i in range(len(corners)): # loop through all corners
         for j in range(i+1, len(corners)):  
            # loop through all of the other corners that we haven;t already looped through
            
            # get two pairs of corners
            corner1 = tuple(corners[i][0])  # as here corners is not flattned so is in the form of [[x,y]]
            # converting to tuples (parameter requirement) as they are list by default
            corner2 = tuple(corners[j][0])

            color = tuple(map(lambda x: int(x), np.random.randint(0,255, size = 3)))   #generate random color, only returns a list, & random ints generated are not all integers (32, 64 bits) when we only want 8bit ints, hence we need to convert them into regular python integers -> map() -> applies the given func to all the element in the arr and returns new arr
            # tuple cast -> parameter req for color
            cv2.line(frame, corner1, corner2, color, 1)

   # showing toggle status 
   status_text = "Lines: ON" if draw_lines else "Lines: OFF"
   cv2.putText(frame, status_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

   cv2.imshow('Live Corner Detection', frame)

   
   key = cv2.waitKey(1) & 0xFF   # gets only the last 8bits ensuring compatibility
   if key == ord('q'):
      break
   elif key == ord('l'):
      draw_lines = not draw_lines

cap.release()
cv2.destroyAllWindows()







   