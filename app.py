import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os
from datetime import datetime
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

st.set_page_config(page_title="Agrojet Smart Scanner", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f1f8e9; }
    .stButton>button { background-color: #2e7d32; color: white; height: 3em; font-size: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌱 Agrojet.ai: Real-time Field Buddy")

# મોડેલ લોડિંગ
@st.cache_resource
def load_agro_model():
    model_path = 'model.tflite'
    if not os.path.exists(model_path): return None
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_agro_model()

if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- PDF Function ---
def generate_pdf(data):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.drawString(100, 750, f"Agrojet.ai Report - {datetime.now().strftime('%Y-%m-%d')}")
    y = 700
    for item in reversed(data[-20:]): # છેલ્લા 20 રેકોર્ડ
        p.drawString(100, y, f"{item['time']} | {item['label']} | Conf: {item['conf']}%")
        y -= 20
    p.save()
    return buffer.getvalue()

# --- Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 Smart Scanner")
    # કેમેરા ઇનપુટ - 'environment' એટલે પાછળનો કેમેરો
    img_file_buffer = st.camera_input("Scan", help="કૃપા કરીને બેક કેમેરો વાપરો", label_visibility="collapsed")

    if img_file_buffer is not None and interpreter:
        img = Image.open(img_file_buffer)
        img_array = np.array(img.convert('RGB'), dtype=np.float32)
        img_resized = cv2.resize(img_array, (224, 224)) / 255.0
        img_reshaped = np.expand_dims(img_resized, axis=0)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], img_reshaped)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        labels = ['PAK (Crop)', 'WEED (Nindan)']
        res_idx = np.argmax(prediction)
        conf = np.max(prediction) * 100
        
        # જામફળને ભૂલથી નીંદણ ના ગણે તે માટે Confidence 85% થી વધુ હોય તો જ WEED ગણવું
        final_label = labels[res_idx]
        if final_label == 'WEED (Nindan)' and conf < 85:
            final_label = 'PAK (Crop)'

        current_time = datetime.now().strftime("%H:%M:%S")
        st.session_state['history'].append({"time": current_time, "label": final_label, "conf": f"{conf:.1f}"})

        if final_label == 'WEED (Nindan)':
            st.error(f"🔴 નીંદણ મળ્યું! ({conf:.1f}%)")
            st.components.v1.html("<script>window.navigator.vibrate(200);</script>")
        else:
            st.success(f"🟢 આ પાક છે. ({conf:.1f}%)")

with col2:
    st.subheader("📜 History Logs")
    if st.session_state['history']:
        pdf_data = generate_pdf(st.session_state['history'])
        st.download_button("📩 Download PDF", data=pdf_data, file_name="agrojet_report.pdf")
        for item in reversed(st.session_state['history'][-10:]):
            st.write(f"*{item['time']}*: {item['label']} ({item['conf']}%)")

if st.button("Clear Logs"):
    st.session_state['history'] = []
    st.rerun()
