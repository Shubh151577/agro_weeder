import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os

# TFLite Runtime ઇમ્પોર્ટ કરવાનો સુરક્ષિત રસ્તો
try:
    import tflite_runtime.interpreter as tflite
    st.sidebar.success("TFLite Runtime લોડ થઈ ગયું છે!")
except ImportError:
    st.error("Error: 'tflite-runtime' ઇન્સ્ટોલ થયું નથી. મહેરબાની કરીને requirements.txt ચેક કરો.")
    st.stop()

st.title("🌱 Agrojet.ai: Live Weeder Detection")

@st.cache_resource
def load_tflite_model():
    model_path = "model.tflite"
    if not os.path.exists(model_path):
        st.error(f"ફાઈલ {model_path} મળી નથી!")
        return None
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img_file_buffer = st.camera_input("પાક કે નીંદણનો ફોટો પાડો")

    if img_file_buffer is not None:
        img = Image.open(img_file_buffer)
        img_array = np.array(img.convert('RGB'), dtype=np.float32)
        img_resized = cv2.resize(img_array, (224, 224)) / 255.0
        img_reshaped = np.expand_dims(img_resized, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_reshaped)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        labels = ['PAK (Crop)', 'WEED (Nindan)']
        result = labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        if result == 'PAK (Crop)':
            st.success(f"✅ આ *પાક (PAK)* છે! ({confidence:.2f}%)")
        else:
            st.error(f"⚠️ આ *નીંદણ (WEED)* છે! ({confidence:.2f}%)")
