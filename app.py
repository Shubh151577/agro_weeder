import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os

st.set_page_config(page_title="Agrojet Live Weeder", layout="centered")
st.title("🌱 Agrojet.ai: Live Weeder Detection")

# મોડેલ લોડ કરવાનું સુરક્ષિત ફંક્શન
@st.cache_resource
def load_tflite_model():
    model_path = "model.tflite"
    if not os.path.exists(model_path):
        st.error(f"મોડેલ ફાઈલ '{model_path}' મળી નથી. કૃપા કરીને GitHub પર અપલોડ કરો.")
        return None
    
    # TensorFlow lite ઇન્ટરપ્રીટર લોડ કરો
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_tflite_model()
    if interpreter:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        img_file_buffer = st.camera_input("પાક કે નીંદણનો ફોટો પાડવા માટે નીચે ક્લિક કરો")

        if img_file_buffer is not None:
            # ફોટો પ્રોસેસિંગ
            img = Image.open(img_file_buffer)
            img_array = np.array(img.convert('RGB'), dtype=np.float32)
            img_resized = cv2.resize(img_array, (224, 224)) / 255.0
            img_reshaped = np.expand_dims(img_resized, axis=0)

            # પ્રેડિક્શન રન કરવું
            interpreter.set_tensor(input_details[0]['index'], img_reshaped)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            
            labels = ['PAK (Crop)', 'WEED (Nindan)']
            result_index = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            st.divider()
            if labels[result_index] == 'PAK (Crop)':
                st.success(f"✅ આ *પાક (PAK)* છે! (વિશ્વાસ: {confidence:.2f}%)")
            else:
                st.error(f"⚠️ આ *નીંદણ (WEED)* છે! (વિશ્વાસ: {confidence:.2f}%)")

except Exception as e:
    st.error(f"સિસ્ટમમાં ભૂલ છે: {e}")
