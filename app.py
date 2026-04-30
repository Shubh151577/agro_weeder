import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
import tensorflow as tf

# RTC સેટિંગ્સ (મોબાઈલ કનેક્શન માટે)
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class VideoTransformer(VideoTransformerBase):
    def _init_(self):
        self.model = tf.lite.Interpreter(model_path="model.tflite")
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
        self.labels = ['PAK (Crop)', 'WEED (Nindan)']

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        # પ્રોસેસિંગ માટે ઈમેજ નાની કરવી
        img_resized = cv2.resize(img, (224, 224)) / 255.0
        img_reshaped = np.expand_dims(img_resized.astype(np.float32), axis=0)

        # પ્રેડિક્શન
        self.model.set_tensor(self.input_details[0]['index'], img_reshaped)
        self.model.invoke()
        prediction = self.model.get_tensor(self.output_details[0]['index'])
        
        result_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        label = self.labels[result_index]

        # બોક્સ અને ટેક્સ્ટનો કલર (Crop માટે લીલો, Weed માટે લાલ)
        color = (0, 255, 0) if label == 'PAK (Crop)' else (0, 0, 255)
        
        # લાઈવ સ્ક્રીન પર બોક્સ બનાવવું
        cv2.rectangle(img, (50, 50), (w-50, h-50), color, 4)
        cv2.putText(img, f"{label} ({confidence:.1f}%)", (60, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        return img

st.title("🌱 Agrojet.ai Live Scanner")
st.write("લાઈવ નીંદણ શોધવા માટે 'Start' બટન દબાવો")

# બેક કેમેરો સેટ કરવા માટે 'facingMode': 'environment'
webrtc_streamer(
    key="agrojet-scanner",
    video_transformer_factory=VideoTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
)

# સાઉન્ડ અને વાઈબ્રેશન માટે નીચે મેસેજ
st.info("નોંધ: નીંદણ દેખાય ત્યારે લાલ બોક્સ અને સાઉન્ડ એલર્ટ આવશે.")
