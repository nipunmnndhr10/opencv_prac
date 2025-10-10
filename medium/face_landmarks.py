import sys
import dlib
import cv2

# path to the pre-trained 68-point facial landmark model.
predictor_model = ".gitignore/shape_predictor_68_face_landmarks.dat"

file_name = sys.argv[1]


image = cv2.imread(file_name)
if image is None:
   print("Error: Unable to load image")
   sys.exit(1)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# creating a HOG face detector
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)

detected_faces = face_detector(image, 1)

print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

for i, face_rect in enumerate(detected_faces):
   x1 = face_rect.left()
   y1 = face_rect.top()
   x2 = face_rect.right()
   y2 = face_rect.bottom()

   print("Face {} found at Left: {}, Top: {}, Right: {}, Bottom{}".format(i,x1,y1,x2,y2))

   # draw boudning box over the detected face
   cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
   
   # Gets the 68 facial landmarks for this face
   landmarks = face_pose_predictor(image, face_rect)

   for j in range(0,68):
      x = landmarks.part(j).x
      y = landmarks.part(j).y
      cv2.circle(image, (x,y), 10, (0,255,255), -1)
   

cv2.imshow("Facial Lanmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



