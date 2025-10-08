import face_recognition
import os

face_image = face_recognition.load_image_file('faces/me.png')

# [0] here means to extract only one face encoding from the image
face_encoding = face_recognition.face_encodings(face_image)[0]

# print(face_image)
print(face_encoding)
# print(os.path.splitext('emma.jpeg')[0])
