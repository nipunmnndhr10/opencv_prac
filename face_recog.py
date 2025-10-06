import face_recognition
import os, sys
import cv2
import numpy as np
import math
import dlib



# calc face acc
def face_confidence(face_distance, face_match_threshold=0.6):
   range = (1.0 - face_match_threshold)
   linear_val = (1.0 - face_distance) / (range * 2.0)

   if face_distance > face_match_threshold:
      return f"{round(linear_val * 100)}%"
   else:
      value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
      return f"{round(value, 2)}%"

class FaceRecognition:
   # class-level variables (shared across all instances of the class)
   face_locations= []
   face_encodings = []
   face_names = []
   known_face_encodings = []
   known_face_names = []
   process_curr_frame = True

   def __init__(self):
      self.encode_faces()

   # func to encode each image in the faces dir
   def encode_faces(self):
      for image in os.listdir('faces'):   # lists all files in the faces directory
         # reads an image file from the path
         face_image = face_recognition.load_image_file(f'faces/{image}') #
         
         # processes the loaded image to detect faces and generate a 128-dimensional face encoding (a numerical vector representing facial features).
         # [0] assumes that each image contains exactly one face. If an image has no faces or multiple faces, this could raise an error: Index Error
         # If multiple faces are detected, only the first face’s encoding is used
         face_encoding = face_recognition.face_encodings(face_image)[0]

 
         # appending the names & encodings of the images
         self.known_face_encodings.append(face_encoding)
         self.known_face_names.append(image)
      
      print(self.known_face_names)

   def run_recognition(self):
      cap = cv2.VideoCapture(0)

      if not cap.isOpened():
         sys.exit('Video source not found...')
      
      while True:
         ret, frame = cap.read()
         
         # process only every second frame
         if self.process_curr_frame:
            # resizing to save computer resources
            small_frame = cv2.resize(frame, (0,0), fx = 0.25, fy = 0.25)
            # select entire img - rows, cols, channels
            # and [::-1] means you're reversing the order of the color channels.
            # rgb_small_frame = small_frame[:, :, ::-1]
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # find all faces in curr frame
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            # self.face_landmarks = face_recognition.face_landmarks(rgb_small_frame)

            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)


            self.face_names = []
            for face_encoding in self.face_encodings:
               # compare known face encodings to the new input face encodings in the curr frame
               # returns a list of booleans (matches) based on a default threshold of 0.6
               matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
               name = 'Unknown'
               confidence = 'Unknown'

               # calculate the similarity ("distance") between a given new face in frame and a list of known faces.

               # Calculates the Euclidean distance (a measure of dissimilarity) between the current face_encoding and each encoding in self.known_face_encodings.
               face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

               # Find the index of the smallest distance (most likely match)
               best_match_index = np.argmin(face_distances)

               # Checks if the closest match (based on distance) is also a valid match in matches
               # if match, return corresponding "labels"
               if matches[best_match_index]:
                  name = self.known_face_names[best_match_index]
                  confidence = face_confidence(face_distances[best_match_index])
            
               self.face_names.append(f'{name} {confidence}')
         
         # to skip a frame, and process only every other frame
         self.process_curr_frame = not self.process_curr_frame

         # display and notations
         for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):

            # bringing img to org dimension
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # print(f"Name: {name}, Location: Top={top}, Right={right}, Bottom={bottom}, Left={left}")

            cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2) 
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,0,255), -1) 
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)
      
         cv2.imshow('Face Recognition', frame)

         if cv2.waitKey(1) == ord('q'):
            break
   
      cap.release()
      cv2.destroyAllWindows()





if __name__ == '__main__':
   fr = FaceRecognition()
   fr.run_recognition()
