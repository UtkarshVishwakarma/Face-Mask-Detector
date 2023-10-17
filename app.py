import streamlit as st
from keras.models import load_model
from PIL import Image
from mtcnn.mtcnn import MTCNN
import os
import cv2

model = load_model('model.h5')

detector = MTCNN()

def save_uploaded_image(image):
    try:
        items = os.listdir('uploads')[0]
        os.remove(os.path.join('uploads', items))

    except:
        pass

    try:
        global img_path
        img_path = os.path.join('uploads', image.name)
        with open(img_path, 'wb') as f:
            f.write(image.getbuffer())

        return True
    
    except:
        return False

st.title("Face Mask Detector 🎭😷")

image = st.file_uploader("Select or Drop a image 👇", type=['png', 'jpeg', 'jpg'])
st.write("⚠ Only! Select or Drop image of type png, jpg and jpeg")

if st.button("Predict"):

    if image is not None:

        if save_uploaded_image(image):

           

            display_image = Image.open(image)
            img = cv2.imread(img_path)

            faces = detector.detect_faces(img)

            x, y, width, height = faces[0]['box']

            img_final = img[x:x+width, y:y+height]
            img_final = cv2.resize(img_final, (224, 224))

            b_img = img_final.reshape(1, 224, 224, 3)

            pred = model.predict(b_img)

            st.image(display_image)

            if pred < 0.50:
                pred = 'No Mask ❌'
                st.warning(pred)

            else:
                pred = 'Mask ✅'
                st.success(pred)

        