import os
import subprocess
import sys

# TensorFlow ને જાતે ઇન્સ્ટોલ કરવાની મેથડ
try:
    import tensorflow as tf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow-cpu"])
    import tensorflow as tf

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import gdown

st.set_page_config(page_title="Agrojet Live Weeder", layout="centered")
st.title("🌱 Agrojet.ai: Live Weeder Detection")

@st.cache_resource
def load_my_model():
    model_path = 'model.h5'
    if not os.path.exists(model_path):
        url = 'https://drive.google.com/uc?id=1t3XF_YyY_S0K_B-5ZqW8V6Xv7XzH0J0L'
        gdown.download(url, model_path, quiet=False)
    model = tf.keras.models.load_model(model_path)
    return model

try:
    model = load_my_model()
    labels = ['PAK (Crop)', 'WEED (Nindan)']
    img_file_buffer = st.camera_input("કેમેરો ચાલુ કરવા માટે નીચે ક્લિક કરો")

    if img_file_buffer is not None:
        img = Image.open(img_file_buffer)
        img_array = np.array(img.convert('RGB'))
        img_resized = cv2.resize(img_array, (224, 224)) / 255.0
        img_reshaped = img_resized.reshape(1, 224, 224, 3)
        prediction = model.predict(img_reshaped)
        result_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        if labels[result_index] == 'PAK (Crop)':
            st.success(f"✅ આ *પાક (PAK)* છે! ({confidence:.2f}%)")
        else:
            st.error(f"⚠️ આ *નીંદણ (WEED)* છે! ({confidence:.2f}%)")
except Exception as e:
    st.error(f"કંઈક ભૂલ થઈ છે: {e}")