import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image

# TFLite લોડ કરવા માટેનો સુરક્ષિત રસ્તો
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow import lite as tflite

# પેજ સેટઅપ
st.set_page_config(page_title="Agrojet Live Weeder", layout="centered")
st.title("🌱 Agrojet.ai: Live Weeder Detection")
st.write("મોડાસા, ગુજરાત - ખેતીમાં ટેકનોલોજીનો સંગમ")

# મોડેલ લોડ કરવાનું ફંક્શન
@st.cache_resource
def load_agro_model():
    model_path = 'model.tflite'
    if not os.path.exists(model_path):
        st.error("મોડેલ ફાઈલ (model.tflite) મળી નથી. મહેરબાની કરીને GitHub પર અપલોડ કરો.")
        return None
    
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# મોડેલ તૈયાર કરવું
interpreter = load_agro_model()

if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # કેમેરા ઇનપુટ
    img_file_buffer = st.camera_input("કેમેરો ચાલુ કરવા માટે બટન દબાવો")

    if img_file_buffer is not None:
        # ઈમેજ પ્રોસેસિંગ
        img = Image.open(img_file_buffer)
        img_array = np.array(img.convert('RGB'), dtype=np.float32)
        
        # મોડેલ મુજબ સાઈઝ સેટ કરવી (224x224)
        img_resized = cv2.resize(img_array, (224, 224)) / 255.0
        img_reshaped = np.expand_dims(img_resized, axis=0)

        # પ્રેડિક્શન કરવું
        interpreter.set_tensor(input_details[0]['index'], img_reshaped)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        # રિઝલ્ટ બતાવવું
        result_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        labels = ['PAK (Crop)', 'WEED (Nindan)']
        final_label = labels[result_index]

        st.divider()
        if final_label == 'PAK (Crop)':
            st.success(f"✅ આ *પાક (PAK)* છે! (વિશ્વાસ: {confidence:.2f}%)")
        else:
            st.error(f"⚠️ આ *નીંદણ (WEED)* છે! (વિશ્વાસ: {confidence:.2f}%)")
        
        st.write("Agrojet.ai - સ્માર્ટ ખેતી, સમૃદ્ધ ખેડૂત.")
