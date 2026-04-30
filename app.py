import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
import tensorflow as tf

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class AgrojetTransformer(VideoTransformerBase):
    def _init_(self):
        self.model = tf.lite.Interpreter(model_path="model.tflite")
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
        self.labels = ['PAK (Crop)', 'WEED (Nindan)']
        self.frame_count = 0 # Frame count rakho

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Darek 5-mi frame par j AI chlavo (Movement smooth krva mate)
        if self.frame_count % 5 == 0:
            small_img = cv2.resize(img, (224, 224))
            input_data = np.expand_dims(small_img.astype(np.float32) / 255.0, axis=0)
            self.model.set_tensor(self.input_details[0]['index'], input_data)
            self.model.invoke()
            prediction = self.model.get_tensor(self.output_details[0]['index'])
            
            res_idx = np.argmax(prediction)
            self.current_label = self.labels[res_idx]
            self.current_conf = np.max(prediction) * 100

        # Boxes hamesha screen pr dekhase
        if hasattr(self, 'current_label'):
            color = (0, 0, 255) if self.current_label == 'WEED (Nindan)' else (0, 255, 0)
            h, w, _ = img.shape
            cv2.rectangle(img, (20, 20), (w-20, h-20), color, 4)
            cv2.putText(img, f"{self.current_label}: {self.current_conf:.1f}%", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return img

st.title("🌱 Agrojet.ai: Real-time Optimized Scanner")

webrtc_streamer(
    key="agrojet-fast",
    video_transformer_factory=AgrojetTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {"facingMode": "environment", "width": 480, "height": 640},
        "audio": False
    },
    async_processing=True, # Background ma AI chlavse jethi video na atkse
)
