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

# પેજ સેટઅપ
st.set_page_config(page_title="Agrojet Smart Weeder", layout="wide")

# મોડેલ લોડ કરવાનું સુરક્ષિત ફંક્શન
@st.cache_resource
def load_agro_model():
    model_path = 'model.tflite'
    if not os.path.exists(model_path):
        st.error(f"મોડેલ ફાઈલ '{model_path}' મળી નથી. કૃપા કરીને GitHub પર અપલોડ કરો.")
        return None
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"મોડેલ લોડ કરવામાં ભૂલ છે: {e}")
        return None

# મોડેલ તૈયાર કરવું
interpreter = load_agro_model()

# હિસ્ટ્રી સેવ કરવા માટે
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- PDF રિપોર્ટ ફંક્શન ---
def generate_pdf(history_data):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # હેડર
    p.setFont("Helvetica-Bold", 18)
    p.drawCentredString(width/2, height - 50, "Agrojet.ai: Smart Weeder Report")
    
    p.setFont("Helvetica", 12)
    p.drawString(50, height - 80, f"જમીન માલિક: રાજેશભાઈ પટેલ")
    p.drawString(50, height - 95, f"સ્થળ: મોડાસા, ગુજરાત")
    p.drawString(50, height - 110, f"તારીખ: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    y = height - 140
    p.line(50, y, width - 50, y)
    y -= 20
    
    # કોલમ નામ
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, y, "Time")
    p.drawString(150, y, "Result")
    p.drawString(300, y, "Confidence")
    p.drawString(400, y, "Notes")
    y -= 10
    p.line(50, y, width - 50, y)
    y -= 20
    
    # હિસ્ટ્રી એડ કરવી (તમારા ઉદાહરણ મુજબ)
    p.setFont("Helvetica", 11)
    for item in reversed(history_data):
        if y < 50:
            p.showPage()
            p.setFont("Helvetica", 11)
            y = height - 50
            
        p.drawString(50, y, item['time'])
        
        # નીંદણ વિગતો
        if item['result'] == "WEED (Nindan)":
            # Red for weed
            p.setFillColorRGB(0.8, 0, 0) 
            p.drawString(150, y, item['result'])
            p.setFillColorRGB(0, 0, 0)
            p.drawString(300, y, f"{item['conf']}%")
            
            # Sthanik & Vaigyanik Naam
            sthanik_naam = "Dungalo (Wild Onion)"
            vaigyanik_naam = "Asphodelus tenuifolius"
            p.drawString(400, y, f"Naam: {sthanik_naam}")
            y -= 15
            p.drawString(400, y, f"V.Naam: {vaigyanik_naam}")
            
        else:
            # Green for crop
            p.setFillColorRGB(0, 0.5, 0)
            p.drawString(150, y, item['result'])
            p.setFillColorRGB(0, 0, 0)
            p.drawString(300, y, f"{item['conf']}%")
            p.drawString(400, y, "Safe Crop")

        y -= 25
        p.line(50, y, 550, y)
        y -= 20
        
    p.save()
    return buffer.getvalue()

# --- મેઈન ઇન્ટરફેસ ---
st.title("🌱 Agrojet.ai: Real-time Smart Scanner")

# layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Scan with Back Camera")
    
    # બેક કેમેરા માટે force-fully 'environment' મોડ
    # તમે switch button નો ઉપયોગ કરીને ફ્રન્ટ પરથી બેક પર જઈ શકશો
    img_file = st.camera_input("Scan your field")

    if img_file is not None and interpreter:
        # ફોટો પ્રોસેસિંગ
        img = Image.open(img_file)
        # ફોટો સેવ કરવો
        
        # AI Processing
        img_array = np.array(img.convert('RGB'), dtype=np.float32)
        img_resized = cv2.resize(img_array, (224, 224)) / 255.0
        img_reshaped = np.expand_dims(img_resized, axis=0)

        # AI Prediction
        i…
