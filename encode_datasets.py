import face_recognition
import os
import pickle

known_persons = {}

for person in os.listdir('./dataset'):
    known_persons[person] = []
    for file_name in os.listdir(os.path.join(f'./dataset/{person}')):
        photo = face_recognition.load_image_file(os.path.join(f'./dataset/{person}/{file_name}'))
        encoded_photo = face_recognition.face_encodings(photo)[0]
        known_persons[person].append(encoded_photo)
with open(os.path.join('./encoded_dataset.pkl'), 'wb') as output_file:
    output_file.write(pickle.dumps(known_persons))