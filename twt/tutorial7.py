import numpy as np
import cv2

# TEMPLATE MATCHING - OBJECT DETECTION

img = cv2.imread('assets/giannis_object.png',0) # loading in grayscale

template = cv2.imread('assets/object-1.png',0)


h,w = template.shape
# print(img)

# methods to try, and select the best one
methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

for method in methods:

   img2 = img.copy()

   # performing convolution -> dragging the template img across the org img to check for similarity in pixel vals
   result = cv2.matchTemplate(img2, template, method)
   # the output of the result will be (W-w+1, H-h+1) where W is width of org img, H is height of org img, w & h are of the template. This is because of the number of times the template can be slided in org images in the x and y axes.

   # return min max val & loc in the result arr
   min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)


   # print(min_loc, max_loc)
   if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
      location = min_loc
      print("min_loc:", min_loc)
   else:
      location = max_loc
      print("max_loc:", max_loc)



   # here we are taking the identified location as top left hand corner of the rectangle to be drawn

   # draw a rectangle (boudning box) on img2 according to location

   # calculating bottom RHC of rectangle
   bottom_right = (location[0] + w, location[1] + h)
   cv2.rectangle(img2, location, bottom_right, 255, 5)
   cv2.imshow("Match",img2)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

