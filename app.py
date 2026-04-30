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

# --- PDF રિપોર્ટ ફંક્શન ---
def generate_pdf(history_data):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    p.setFont("Helvetica-Bold", 16)
    p.drawCentredString(width/2, height - 50, "Agrojet.ai: Detailed Field Report")
    
    y = height - 100
    for item in reversed(history_data):
        if y < 150:
            p.showPage()
            y = height - 50
        
        p.setFont("Helvetica-Bold", 11)
        p.drawString(50, y, f"Time: {item['time']}")
        
        if "WEED" in item['label']:
            p.setFillColorRGB(0.8, 0, 0)
            p.drawString(50, y - 15, "Result: Nindan (Weed) Detected")
            p.setFillColorRGB(0, 0, 0)
            p.setFont("Helvetica", 10)
            p.drawString(50, y - 30, "Sthanik Naam: Dungalo (Wild Onion)")
            p.drawString(50, y - 45, "Vaigyanik Naam: Asphodelus tenuifolius")
        else:
            p.setFillColorRGB(0, 0.5, 0)
            p.drawString(50, y - 15, "Result: Surakshit Pak (Safe)")
            p.setFillColorRGB(0, 0, 0)
            p.setFont("Helvetica", 10)
            p.drawString(50, y - 30, "Notes: Area is clean and safe.")

        p.drawString(350, y - 15, f"Confidence: {item['conf']}%")
        y -= 70
        p.line(50, y + 5, 550, y + 5)
    p.save()
    return buffer.getvalue()

# --- App Interface ---
st.title("🌱 Agrojet.ai: Advanced Weeder")

# HTML/JS - આનાથી સીધો બેક કેમેરો ખુલશે
st.subheader("📸 Scan with Back Camera")
img_file = st.camera_input("Capture Photo", label_visibility="collapsed")

# જો ઉપરના બટનથી ના થાય, તો આ વધારાની ટીપ
st.info("💡 જો ફ્રન્ટ કેમેરો ખુલે, તો કેમેરાની ઉપર રહેલા 'Switch' આઈકન પર ક્લિક કરો.")

if img_file and interpreter:
    img = Image.open(img_file)
    img_array = np.array(img.convert('RGB'), dtype=np.float32)
    img_resized = cv2.resize(img_array, (224, 224)) / 255.0
    img_reshaped = np.expand_dims(img_resized, axis=0)

    # Prediction
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_reshaped)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    labels = ['PAK (Crop)', 'WEED (Nindan)']
    res_idx = np.argmax(prediction)
    conf = np.max(prediction) * 100
    
    # ૮૫% ની ઉપર હોય તો જ નીંદણ ગણવું (તમારા જામફળને બચાવવા માટે)
    final_label = labels[res_idx]
    if final_label == 'WEED (Nindan)' and conf < 85:
        final_label = 'PAK (Crop)'

    current_time = datetime.now().strftime("%I:%M %p")
    st.session_state['history'].append({
        "time": current_time, 
        "label": final_label, 
        "conf": f"{conf:.1f}",
        "image": img
    })

    # Display Result
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(img, caption="Scanned Image", use_container_width=True)
    with col_b:
        if "WEED" in final_label:
            st.error(f"🔴 {final_label} Detected! \nConfidence: {conf:.1f}%")
            st.components.v1.html("<script>window.navigator.vibrate(200);</script>")
        else:
            st.success(f"🟢 {final_label} Safe. \nConfidence: {conf:.1f}%")

# --- History Section ---
st.divider()
st.subheader("📜 History & Detailed Report")

if st.session_state['history']:
    pdf_data = generate_pdf(st.session_state['history'])
    st.download_button("📩 Download PDF Report", data=pdf_data, file_name="Agrojet_Report.pdf")
    
    for item in reversed(st.session_state['history'][-10:]):
        with st.expander(f"{item['time']} - {item['label']} ({item['conf']}%)"):
            st.image(item['image'], width=200)
            if "WEED" in item['label']:
                st.write("Sthanik Naam: Dungalo (Wild Onion)")
                st.write("Vaigyanik Naam: Asphodelus tenuifolius")
else:
    st.info("કોઈ ડેટા નથી. ખેતરમાં જઈને સ્કેન શરૂ કરો.")

if st.button("🗑️ Clear All Logs"):
    st.session_state['history'] = []
    st.rerun()
