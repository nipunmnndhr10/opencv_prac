import face_recognition

face_image = face_recognition.load_image_file('faces/me.png')

face_encoding = face_recognition.face_encodings(face_image)[0]

print(face_encoding)