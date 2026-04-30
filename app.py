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

st.set_page_config(page_title="Agrojet Smart Weeder", layout="wide")

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

# --- PDF રિપોર્ટ (તમારા એક્ઝામ્પલ મુજબ) ---
def generate_pdf(history_data):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    p.setFont("Helvetica-Bold", 18)
    p.drawCentredString(width/2, height - 50, "Agrojet.ai: Smart Weeder Report")
    
    p.setFont("Helvetica", 12)
    p.drawString(50, height - 80, f"Location: Modasa, Gujarat")
    p.drawString(50, height - 95, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    y = height - 130
    p.line(50, y, width-50, y)
    y -= 30

    for item in reversed(history_data):
        if y < 150:
            p.showPage()
            y = height - 50
        
        p.setFont("Helvetica-Bold", 12)
        p.drawString(50, y, f"Time: {item['time']}")
        
        if "WEED" in item['label']:
            p.setFillColorRGB(0.8, 0, 0) # Red
            p.drawString(180, y, f"Result: {item['label']}")
            p.setFillColorRGB(0, 0, 0)
            p.setFont("Helvetica", 10)
            p.drawString(50, y - 20, "Sthanik Naam: Dungalo (Wild Onion)")
            p.drawString(50, y - 35, "Vaigyanik Naam: Asphodelus tenuifolius")
        else:
            p.setFillColorRGB(0, 0.5, 0) # Green
            p.drawString(180, y, f"Result: {item['label']}")
            p.setFillColorRGB(0, 0, 0)
            p.setFont("Helvetica", 10)
            p.drawString(50, y - 20, "Status: Safe Crop (Jamfal/Other)")
            
        p.drawString(350, y, f"Confidence: {item['conf']}%")
        y -= 60
        p.line(50, y, 550, y)
        y -= 30
    
    p.save()
    return buffer.getvalue()

# --- Main App Interface ---
st.title("🌱 Agrojet.ai: Smart Scanner")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📸 Scan with Back Camera")
    
    # સીધો બેક કેમેરો ખોલવા માટે આ બટન બેસ્ટ છે
    img_file = st.file_uploader("Click here to Open Camera", type=['jpg', 'jpeg', 'png'])
    
    st.info("💡 'Browse' પર ક્લિક કરીને 'Camera' સિલેક્ટ કરો. તેનાથી તમારા ફોનનો અસલી બેક કેમેરો ખુલશે.")

    if img_file and interpreter:
        img = Image.open(img_file)
        # Display Image
        st.image(img, caption="Scanned Image", use_container_width=True)
        
        # AI Processing
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
        
        # Detection Correction: 85% Confidence Logic
        final_label = labels[res_idx]
        if final_label == 'WEED (Nindan)' and conf < 85:
            final_label = 'PAK (Crop)'

        current_time = datetime.now().strftime("%I:%M %p")
        
        # હિસ્ટ્રીમાં સેવ કરવું (ફોટા સાથે)
        st.session_state['history'].append({
            "time": current_time, 
            "label": final_label, 
            "conf": f"{conf:.1f}",
            "image": img
        })

        if "WEED" in final_label:
            st.error(f"🚨 {final_label} Detected! ({conf:.1f}%)")
            st.components.v1.html("<script>window.navigator.vibrate(200);</script>")
        else:
            st.success(f"✅ {final_label} - Field is Safe. ({conf:.1f}%)")

with col2:
    st.subheader("📜 History & Report")
    if st.session_state['history']:
        pdf_data = generate_pdf(st.session_state['history'])
        st.download_button("📥 Download PDF Report", data=pdf_data, file_name="Agrojet_Report.pdf")
        
        # હિસ્ટ્રી લિસ્ટ
        for item in reversed(st.session_state['history'][-10:]):
            with st.expander(f"{item['time']} - {item['label']}"):
                st.image(item['image'], width=150)
                st.write(f"Confidence: {item['conf']}%")
                if "WEED" in item['label']:
                    st.write("Sthanik Naam: Dungalo")
    else:
        st.info("હજુ સુધી કોઈ સ્કેનિંગ કર્યું નથી.")

if st.button("🗑️ Clear All Logs"):
    st.session_state['history'] = []
    st.rerun()
