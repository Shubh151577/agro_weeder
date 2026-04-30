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

# --- CSS for Professional Look ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; background-color: #4CAF50; color: white; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌱 Agrojet.ai: Smart Weeder & History")

# મોડેલ લોડિંગ
@st.cache_resource
def load_agro_model():
    model_path = 'model.tflite'
    if not os.path.exists(model_path): return None
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_agro_model()

# હિસ્ટ્રી સ્ટોર કરવા માટે session_state
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- PDF બનાવવાનું ફંક્શન ---
def generate_pdf(data):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, 750, "Agrojet.ai - Field Scan Report")
    p.setFont("Helvetica", 12)
    p.drawString(100, 730, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    y = 700
    p.drawString(100, y, "Time | Label | Confidence")
    y -= 20
    p.line(100, y, 500, y)
    y -= 20

    for item in reversed(data):
        if y < 50: # નવી પેજ માટે
            p.showPage()
            y = 750
        text = f"{item['time']} | {item['label']} | {item['conf']}%"
        p.drawString(100, y, text)
        y -= 20
        
    p.save()
    return buffer.getvalue()

# --- મેઈન લેઆઉટ ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Live Scan")
    img_file_buffer = st.camera_input("Scan Field")

    if img_file_buffer is not None and interpreter:
        # AI Processing
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
        label = labels[res_idx]
        current_time = datetime.now().strftime("%H:%M:%S")

        # હિસ્ટ્રીમાં એડ કરવું
        st.session_state['history'].append({
            "time": current_time,
            "label": label,
            "conf": f"{conf:.1f}"
        })

        if label == 'WEED (Nindan)':
            st.error(f"⚠️ {label} મળ્યું! ({conf:.1f}%)")
            st.components.v1.html("<script>window.navigator.vibrate(500);</script>")
        else:
            st.success(f"✅ {label} ({conf:.1f}%)")

with col2:
    st.subheader("📜 History Logs")
    if st.session_state['history']:
        # PDF ડાઉનલોડ બટન
        pdf_data = generate_pdf(st.session_state['history'])
        st.download_button(label="📥 Download PDF Report", data=pdf_data, 
                           file_name="agrojet_report.pdf", mime="application/pdf")
        
        # હિસ્ટ્રી ટેબલ
        for item in reversed(st.session_state['history']):
            color = "🔴" if "WEED" in item['label'] else "🟢"
            st.write(f"{color} *{item['time']}* - {item['label']} ({item['conf']}%)")
    else:
        st.info("હજુ સુધી કોઈ સ્કેનિંગ કર્યું નથી.")

if st.button("🗑️ Clear History"):
    st.session_state['history'] = []
    st.rerun()
