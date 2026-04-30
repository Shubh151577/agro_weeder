import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tflite_runtime.interpreter as tflite

st.title("🌱 Agrojet.ai: Live Weeder Detection")

@st.cache_resource
def load_tflite_model():
    # Folder mathi sidhu nanu model load karo
    interpreter = tflite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    img = Image.open(img_file_buffer)
    img_array = np.array(img.convert('RGB'), dtype=np.float32)
    img_resized = cv2.resize(img_array, (224, 224)) / 255.0
    img_reshaped = np.expand_dims(img_resized, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_reshaped)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    if np.argmax(prediction) == 0:
        st.success("✅ Aa PAK (Crop) che!")
    else:
        st.error("⚠️ Aa WEED (Nindan) che!")
