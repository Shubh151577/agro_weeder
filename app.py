import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
import tensorflow as tf

# RTC સેટિંગ્સ - મોબાઈલ માટે સ્ટેબલ કનેક્શન
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class AgrojetTransformer(VideoTransformerBase):
    def _init_(self):
        # મોડેલ એક જ વાર લોડ થશે
        self.model = tf.lite.Interpreter(model_path="model.tflite")
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
        self.labels = ['PAK (Crop)', 'WEED (Nindan)']

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # સ્પીડ વધારવા માટે ફ્રેમની સાઈઝ ઘટાડવી
        small_img = cv2.resize(img, (224, 224))
        input_data = np.expand_dims(small_img.astype(np.float32) / 255.0, axis=0)

        # AI પ્રેડિક્શન
        self.model.set_tensor(self.input_details[0]['index'], input_data)
        self.model.invoke()
        prediction = self.model.get_tensor(self.output_details[0]['index'])
        
        res_idx = np.argmax(prediction)
        conf = np.max(prediction) * 100
        label = self.labels[res_idx]

        # કલર અને બોક્સ (ગ્રીન - પાક, રેડ - નીંદણ)
        color = (0, 0, 255) if label == 'WEED (Nindan)' else (0, 255, 0)
        
        # ઓવરલે (ડ્રોઈંગ)
        h, w, _ = img.shape
        cv2.rectangle(img, (10, 10), (w-10, h-10), color, 4)
        cv2.putText(img, f"{label}: {conf:.1f}%", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return img

st.title("🌱 Agrojet.ai: Real-time Field Scanner")

webrtc_streamer(
    key="agrojet-live",
    video_transformer_factory=AgrojetTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {
            "facingMode": "environment",
            "frameRate": {"ideal": 10, "max": 15} # ફ્રેમ રેટ કંટ્રોલ કરવાથી હેંગ નહીં થાય
        },
        "audio": False
    },
    async_processing=True, # બેકગ્રાઉન્ડ પ્રોસેસિંગ ચાલુ કરવું
)

st.warning("નોંધ: જો વિડિયો હેંગ થાય, તો એકવાર પેજ Refresh કરીને ફરી 'Start' દબાવો.")
