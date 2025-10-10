import sys
import dlib
import cv2

# Detecting (detection not recognition) faces using the dlib library’s Histogram of Oriented Gradients (HOG) face detector
#           no encoding or matching

# take image file name from cmd line 
if len(sys.argv) < 2:
   print("Error: Please provide an image file path as a cmd-line arg")
   sys.exit(1)  # stops your Python program and exits, stopping execution early

file_name = sys.argv[1]



# create HOG face detector
face_detector = dlib.get_frontal_face_detector()

# print(dir(dlib))  # List all the attributes and methods of dlib

# Check for the specific method
# print("get_frontal_face_detector" in dir(dlib))


# load image into an array
image = cv2.imread(file_name)
if image is None:
   print(f"Error: Could not load image {file_name}")
   sys.exit(1)


# run HOG face detector on the img
# returs list of dlib.rectangle objects
# result = bounding box on faces
detected_faces = face_detector(image, 1)

print(detected_faces)

print("Found {} faces in the file {}".format(len(detected_faces), file_name))



# loop through each face found in img
# Loops over dlib.rectangle objects with indices (i).
for i, face_rect in enumerate(detected_faces):
   x1 = face_rect.left()
   y1= face_rect.top()
   x2= face_rect.right()
   y2 =face_rect.bottom()
   print("Face #{} found at Left: {}, Top: {}, Right: {}, Bottom: {}".format(i, x1,y1,x2,y2))
   cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
               # top-left coor,  bottom-right coordinates


cv2.imshow("Detected faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
