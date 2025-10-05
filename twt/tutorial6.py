import numpy as np
import cv2

# edge, corner detection


img = cv2.imread('assets/chessboard.png')


# grayscale conversion
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# shi-tomasi corner detection algo
corners = cv2.goodFeaturesToTrack(gray, 5, 0.5, 100)   # src img, num of corners. min quality, min euclidean distance betn corners

# convert corners into int
corners = np.intp(corners)
# print(corners) # default are floating point vals

for corner in corners:  # gives an 2d array
   # print(corner)
   x, y = corner.ravel()   # get one corner coordinates, .ravel() flattens an array
   cv2.circle(img, (x,y), 10, (255,127,0), -1)  # drawing circles in the corner coordinates


# draw random colored lines between every corners

for i in range(len(corners)): # loop through all corners
   for j in range(i+1, len(corners)):  
      # loop through all of the other corners that we haven;t already looped through
      
      # get two pairs of corners
      corner1 = tuple(corners[i][0])  # as here corners is not flattned so is in the form of [[x,y]]
      # converting to tuples (parameter requirement) as they are list by default
      corner2 = tuple(corners[j][0])

      color = tuple(map(lambda x: int(x), np.random.randint(0,255, size = 3)))   #generate random color, only returns a list, & random ints generated are not all integers (32, 64 bits) when we only want 8bit ints, hence we need to convert them into regular python integers -> map() -> applies the given func to all the element in the arr and returns new arr
      # tuple cast -> parameter req for color
      cv2.line(img, corner1, corner2, color, 1)




cv2.imshow('Chessboard', img)
cv2.waitKey(0)
cv2.destroyAllWindows()