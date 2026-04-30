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

# --- એડવાન્સ PDF રિપોર્ટ (તમારા એક્ઝામ્પલ મુજબ) ---
def generate_detailed_pdf(history_data):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height - 50, "Agrojet.ai: Smart Weeder Report")
    
    y_position = height - 100
    
    for item in reversed(history_data[-10:]): # છેલ્લા 10 રેકોર્ડ
        if y_position < 150:
            c.showPage()
            y_position = height - 50
        
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_position, f"Time: {item['time']}")
        
        # નીંદણની વિગતો (તમારા ફોટા મુજબ)
        c.setFont("Helvetica", 11)
        if "WEED" in item['label']:
            c.setFillColorRGB(0.8, 0, 0) # Red for weed
            c.drawString(50, y_position - 20, "Sthanik Naam: Dungalo (Wild Onion)")
            c.drawString(50, y_position - 35, "Vaigyanik Naam: Asphodelus tenuifolius")
            c.drawString(50, y_position - 50, f"Confidence: {item['conf']}%")
        else:
            c.setFillColorRGB(0, 0.5, 0) # Green for crop
            c.drawString(50, y_position - 20, "Label: Surakshit Pak (Safe Crop)")
            c.drawString(50, y_position - 35, f"Confidence: {item['conf']}%")
        
        c.setFillColorRGB(0, 0, 0)
        c.line(50, y_position - 60, 550, y_position - 60)
        y_position -= 80
        
    c.save()
    return buffer.getvalue()

# --- Main Interface ---
st.title("🌱 Agrojet.ai: Smart Field Buddy")

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("📷 Scan Field")
    # Back Camera માટે 'environment' સેટિંગ
    img_file = st.camera_input("Take Photo", help="Use back camera for better results")

    if img_file and interpreter:
        img = Image.open(img_file)
        img_array = np.array(img.convert('RGB'), dtype=np.float32)
        img_resized = cv2.resize(img_array, (224, 224)) / 255.0
        img_reshaped = np.expand_dims(img_resized, axis=0)

        # AI Prediction
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], img_reshaped)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        labels = ['PAK (Crop)', 'WEED (Nindan)']
        res_idx = np.argmax(prediction)
        conf = np.max(prediction) * 100
        
        # Jamfal (Crop) ને બચાવવા માટે 85% Confidence લોજિક
        label = labels[res_idx]
        if label == 'WEED (Nindan)' and conf < 85:
            label = 'PAK (Crop)'

        current_time = datetime.now().strftime("%I:%M %p")
        st.session_state['history'].append({"time": current_time, "label": label, "conf": f"{conf:.1f}"})

        if label == 'WEED (Nindan)':
            st.error(f"🔴 {label} Detected! ({conf:.1f}%)")
        else:
            st.success(f"🟢 {label} Safe. ({conf:.1f}%)")

with col2:
    st.subheader("📜 History & Report")
    if st.session_state['history']:
        pdf = generate_detailed_pdf(st.session_state['history'])
        st.download_button("📥 Download Detailed PDF", data=pdf, file_name="Agrojet_Report.pdf")
        
        for item in reversed(st.session_state['history']):
            st.write(f"*{item['time']}* - {item['label']} ({item['conf']}%)")
    else:
        st.info("No scans yet.")
