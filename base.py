from flask import Flask, request, Response
import jsonpickle
import face_recognition
import cv2
import numpy as np


chirag_image = face_recognition.load_image_file("assets/chirag.jpeg")
chirag_face_encoding = face_recognition.face_encodings(chirag_image)[0]

known_face_encodings = [chirag_face_encoding]
known_face_names = ["Chirag"]


app = Flask(__name__)

@app.route('/')
def index():
   return "This is a very beautiful link"

@app.route('/imageprocess',methods=['POST'])
def imageprocess():
   r = request
   nparr = np.fromstring(r.data, np.uint8)
   print(nparr)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
   rgb_small_frame = small_frame[:, :, ::-1]
   face_locations = face_recognition.face_locations(rgb_small_frame)
   face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
   face_names = []
   for face_encoding in face_encodings:
      matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
      face_distances = face_recognition.face_distance(known_face_encodings,face_encoding)   
      best_match_index = np.argmin(face_distances)
      if matches[best_match_index]:
         name = known_face_names[best_match_index]
         face_names.append(name)
         print("Face names",face_names)
   response = {'message': 'image received. size= and the user matched is {}'.format(str(face_names[0]))}
   response_pickled = jsonpickle.encode(response)
   print(response)
   return Response(response=response_pickled, status=200, mimetype="application/json")


app.run(host="0.0.0.0", port=5000,debug=True)
