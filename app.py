import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time

st.set_page_config(page_title="Agrojet.ai Live", layout="centered")
st.title("🌱 Agrojet.ai: Real-time Smart Scanner")

# મોડેલ લોડ કરો
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# વિડિયો સ્ટ્રીમિંગ માટે સાદો અને ફાસ્ટ રસ્તો
img_file_buffer = st.camera_input("Scan your field")

if img_file_buffer is not None:
    # પ્રોસેસિંગ સ્પીડ વધારવા માટે
    img = Image.open(img_file_buffer)
    img_array = np.array(img.convert('RGB'), dtype=np.float32)
    
    # AI ડિટેક્શન
    img_resized = cv2.resize(img_array, (224, 224)) / 255.0
    img_reshaped = np.expand_dims(img_resized, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_reshaped)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    labels = ['PAK (Crop)', 'WEED (Nindan)']
    res_idx = np.argmax(prediction)
    conf = np.max(prediction) * 100

    # રિઝલ્ટ મુજબ વાઈબ્રેશન અને સાઉન્ડ
    if labels[res_idx] == 'WEED (Nindan)' and conf > 75:
        st.error(f"🚨 નીંદણ મળ્યું! ({conf:.1f}%)")
        # મોબાઈલ વાઈબ્રેશન માટે JS
        st.components.v1.html("<script>window.navigator.vibrate(500);</script>")
    else:
        st.success(f"✅ આ પાક છે. ({conf:.1f}%)")
