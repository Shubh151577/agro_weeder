import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
import tensorflow as tf

# Google STUN સર્વર કનેક્શન માટે
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class AgrojetTransformer(VideoTransformerBase):
    def _init_(self):
        self.model = tf.lite.Interpreter(model_path="model.tflite")
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
        self.labels = ['PAK (Crop)', 'WEED (Nindan)']

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        # AI પ્રોસેસિંગ
        img_resized = cv2.resize(img, (224, 224)) / 255.0
        img_reshaped = np.expand_dims(img_resized.astype(np.float32), axis=0)

        self.model.set_tensor(self.input_details[0]['index'], img_reshaped)
        self.model.invoke()
        prediction = self.model.get_tensor(self.output_details[0]['index'])
        
        res_idx = np.argmax(prediction)
        conf = np.max(prediction) * 100
        label = self.labels[res_idx]

        # કલર સેટિંગ: Weed માટે લાલ (Red), Crop માટે લીલો (Green)
        color = (0, 0, 255) if label == 'WEED (Nindan)' else (0, 255, 0)
        
        # લાઈવ બોક્સ અને ટેક્સ્ટ
        cv2.rectangle(img, (20, 20), (w-20, h-20), color, 6)
        cv2.putText(img, f"Agrojet: {label} ({conf:.1f}%)", (30, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        return img

st.title("🌱 Agrojet.ai: Live Field Scanner")
st.write("નીંદણ શોધવા માટે 'Start' બટન દબાવો")

# સાઉન્ડ માટેનું સેટિંગ
weed_alert_html = """
<audio id="weed_audio" src="https://www.soundjay.com/buttons/beep-01a.mp3" preload="auto"></audio>
<script>
function playWeedSound() {
    document.getElementById('weed_audio').play();
}
</script>
"""
st.components.v1.html(weed_alert_html, height=0)

# લાઈવ વિડિયો સ્ટ્રીમર
webrtc_streamer(
    key="agrojet-scanner",
    video_transformer_factory=AgrojetTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {"facingMode": "environment"}, # આનાથી બેક કેમેરો ખુલશે
        "audio": False
    },
)

st.info("💡 ટિપ: ખેતરમાં સ્કેનિંગ કરતી વખતે લાલ બોક્સ દેખાય તો ત્યાં નીંદણ છે.")
