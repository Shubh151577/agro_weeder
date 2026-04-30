import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="Agrojet Live Weeder", layout="centered")

st.title("🌱 Agrojet.ai: Live Weeder Detection")
st.write("જામફળ, ગલગોટા (PAK) અને નીંદણ (WEED) ની લાઈવ ઓળખ")

# મોડેલ લોડ કરવું
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model('model.h5')
    return model

model = load_my_model()
labels = ['PAK (Crop)', 'WEED (Nindan)']

# મોબાઈલ કેમેરા માટે સેટિંગ
img_file_buffer = st.camera_input("કેમેરો ચાલુ કરવા માટે નીચે ક્લિક કરો")

if img_file_buffer is not None:
    # ફોટોને પ્રોસેસ કરવો
    img = Image.open(img_file_buffer)
    img_array = np.array(img.convert('RGB'))
    img_resized = cv2.resize(img_array, (224, 224)) / 255.0
    img_reshaped = img_resized.reshape(1, 224, 224, 3)

    # Prediction
    prediction = model.predict(img_reshaped)
    result_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # રિઝલ્ટ બતાવવું
    if labels[result_index] == 'PAK (Crop)':
        st.success(f"✅ આ *પાક (PAK)* છે! (ચોકસાઈ: {confidence:.2f}%)")
    else:
        st.error(f"⚠️ આ *નીંદણ (WEED)* છે! (ચોકસાઈ: {confidence:.2f}%)")