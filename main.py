import cv2
import face_recognition
import pickle
import numpy as np
import streamlit as st
encoding_with_ids = pickle.load(open('encoding_with_ids.pkl', 'rb'))
encodings, person_names = encoding_with_ids

test_image = 'test.jpg'



def main():
    st.title('Face Recognition Project by Noor ul Eman')
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Process'):
            image = cv2.imread(test_image)
            resized_image = cv2.resize(image, (0, 0), None, 0.25, 0.25)
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

            curr_face_location = face_recognition.face_locations(resized_image)
            curr_face_encodings = face_recognition.face_encodings(resized_image, curr_face_location)

            for cfe, cfl in zip(curr_face_encodings, curr_face_location):
                matched_faces = face_recognition.compare_faces(encodings, cfe)
                face_similarity_distance = face_recognition.face_distance(encodings, cfe)
                match_index = np.argmin(face_similarity_distance)
                
                if matched_faces[match_index]:
                    st.write(f"Match found! Person is {person_names[match_index]}")
                else:
                    st.warning("No matches found.")

if __name__=='__main__':
    main()
