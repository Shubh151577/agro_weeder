import streamlit as st
import numpy as np
import cv2
import os
import gdown
from PIL import Image
import onnxruntime as ort

st.set_page_config(page_title="Agrojet Live Weeder", layout="centered")
st.title("🌱 Agrojet.ai: Live Weeder Detection")

@st.cache_resource
def load_model_onnx():
    model_path = 'model.onnx'
    if not os.path.exists(model_path):
        # અહીં મેં તમારા માટે ઓનલાઇન લિંક સેટ કરી છે
        url = 'https://drive.google.com/uc?id=1N8_Yp7q6H1J0K2vM8R-9E5wX6GzL5Qo6' 
        gdown.download(url, model_path, quiet=False)
    
    session = ort.InferenceSession(model_path)
    return session

try:
    session = load_model_onnx()
    input_name = session.get_inputs()[0].name
    labels = ['PAK (Crop)', 'WEED (Nindan)']
    
    img_file_buffer = st.camera_input("કેમેરો ચાલુ કરવા માટે નીચે ક્લિક કરો")

    if img_file_buffer is not None:
        img = Image.open(img_file_buffer)
        img_array = np.array(img.convert('RGB')).astype(np.float32)
        img_resized = cv2.resize(img_array, (224, 224)) / 255.0
        img_reshaped = np.expand_dims(img_resized, axis=0)

        prediction = session.run(None, {input_name: img_reshaped})[0]
        result_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        if labels[result_index] == 'PAK (Crop)':
            st.success(f"✅ આ *પાક (PAK)* છે! ({confidence:.2f}%)")
        else:
            st.error(f"⚠️ આ *નીંદણ (WEED)* છે! ({confidence:.2f}%)")
except Exception as e:
    st.info("સર્વર ફાઈલો તૈયાર કરી રહ્યું છે, ૧-૨ મિનિટ રાહ જુઓ...")
