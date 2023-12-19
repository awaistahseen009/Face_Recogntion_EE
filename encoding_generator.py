import cv2 as cv
import pickle
import face_recognition
import os
person_images=list()
person_names=list()
folder_path='Images/'
for image in os.listdir(folder_path):
    person_images.append(cv.imread(os.path.join(folder_path, image)))
    person_names.append(os.path.splitext(image)[0])

def encode_image(image_list):
    encoded_images=list()
    for img in image_list:
        img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        encod_image=face_recognition.face_encodings(img)[0]
        encoded_images.append(encod_image)
    return encoded_images


encodings = encode_image(person_images)
encoding_with_ids=[
    encodings,
    person_names
]
pickle.dump(encoding_with_ids,open('encoding_with_ids.pkl','wb'))
