import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os

# પેજ સેટઅપ
st.set_page_config(page_title="Agrojet Smart Scanner", layout="centered")
st.title("🌱 Agrojet.ai: Smart Weeder")

# મોડેલ લોડિંગ
@st.cache_resource
def load_agro_model():
    model_path = 'model.tflite'
    if not os.path.exists(model_path):
        st.error("મોડેલ ફાઈલ મળી નથી!")
        return None
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_agro_model()

if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # કેમેરા ઇનપુટ (આ હેંગ નહીં થાય)
    img_file_buffer = st.camera_input("ખેતર સ્કેન કરવા માટે ફોટો પાડો")

    if img_file_buffer is not None:
        # ઈમેજ પ્રોસેસિંગ
        img = Image.open(img_file_buffer)
        img_array = np.array(img.convert('RGB'), dtype=np.float32)
        
        # સાઈઝ સેટ કરવી
        img_resized = cv2.resize(img_array, (224, 224)) / 255.0
        img_reshaped = np.expand_dims(img_resized, axis=0)

        # AI પ્રેડિક્શન
        interpreter.set_tensor(input_details[0]['index'], img_reshaped)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        result_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        labels = ['PAK (Crop)', 'WEED (Nindan)']
        final_label = labels[result_index]

        st.divider()
        if final_label == 'WEED (Nindan)' and confidence > 70:
            st.error(f"⚠️ નીંદણ (WEED) મળ્યું! ({confidence:.1f}%)")
            # મોબાઈલ વાઈબ્રેશન
            st.components.v1.html("<script>window.navigator.vibrate(500);</script>")
        else:
            st.success(f"✅ પાક (PAK) સુરક્ષિત છે. ({confidence:.1f}%)")

st.info("💡 લાઈવ સ્કેનિંગમાં હેંગ થવાની સમસ્યાને કારણે આ પદ્ધતિ સૌથી વધુ સ્ટેબલ છે.")
