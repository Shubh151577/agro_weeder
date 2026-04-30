import streamlit as st
import numpy as np
import cv2
import os
import gdown
from PIL import Image
import tflite_runtime.interpreter as tflite

st.set_page_config(page_title="Agrojet Live Weeder", layout="centered")
st.title("🌱 Agrojet.ai: Live Weeder Detection")

@st.cache_resource
def load_tflite_model():
    model_path = 'model.tflite'
    if not os.path.exists(model_path):
        # અહીં તમારા TFLite મોડેલની લિંક આવશે
        url = 'https://drive.google.com/uc?id=1t3XF_YyY_S0K_B-5ZqW8V6Xv7XzH0J0L' 
        gdown.download(url, model_path, quiet=False)
    
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_tflite_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    labels = ['PAK (Crop)', 'WEED (Nindan)']
    img_file_buffer = st.camera_input("કેમેરો ચાલુ કરવા માટે નીચે ક્લિક કરો")

    if img_file_buffer is not None:
        img = Image.open(img_file_buffer)
        img_array = np.array(img.convert('RGB'), dtype=np.float32)
        img_resized = cv2.resize(img_array, (224, 224)) / 255.0
        img_reshaped = np.expand_dims(img_resized, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_reshaped)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        result_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        if labels[result_index] == 'PAK (Crop)':
            st.success(f"✅ આ *પાક (PAK)* છે! ({confidence:.2f}%)")
        else:
            st.error(f"⚠️ આ *નીંદણ (WEED)* છે! ({confidence:.2f}%)")
except Exception as e:
    st.info("સર્વર તૈયાર થઈ રહ્યું છે, મહેરબાની કરીને થોડી રાહ જુઓ...")
