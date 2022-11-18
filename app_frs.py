from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import face_recognition
import streamlit as st
from PIL import Image
import numpy as np

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
st.title(' Face Recognition System')
labels_placeholder = st.empty()

st.subheader("Upload Your Photo")


# if st.button('Upload Image'):
uploaded_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg'])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    data = Image.fromarray(image)
    data.save('test.png')

st.subheader("Test it!!")
class VideoProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        faces = cascade.detectMultiScale(frm, 1.1, 3)
        for x,y,w,h in faces:
            img = cv2.cvtColor(frm[y:y+h,x:x+w],cv2.COLOR_BGR2RGB)
            cv2.rectangle(frm, (x,y), (x+w, y+h), (0,255,0), 3)
            if len(face_recognition.face_encodings(img)) == 0:
                continue
            else:     
                img_encode = face_recognition.face_encodings(img)[0] 
                print(img_encode)
                
                known_image = face_recognition.load_image_file('test.png')
                encoding = face_recognition.face_encodings(known_image)[0]
                result = face_recognition.compare_faces([encoding],img_encode,0.6)
                if result[0] == True:
                    cv2.putText(frm,"Matched",(x-8,y),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
                    cv2.rectangle(frm,(x,y),(x+w,y+h),(0,255,0),2)  
                else:
                    cv2.putText(frm,"Not Matched",(x-8,y),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
                    cv2.rectangle(frm,(x,y),(x+w,y+h),(0,0,255),2)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


webrtc_ctx = webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
)
