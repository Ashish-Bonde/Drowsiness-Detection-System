import streamlit as st
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import pygame
import tempfile


# pygame.mixer.init() 'for streamlit cloud it creates problem'

pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.init()


facemodel=cv2.CascadeClassifier('face.xml')
eyesmodel=load_model('eye.h5')
yawnmodel=load_model('yawn.h5')
drowsymodel=load_model('drowsy.h5')
eyexml=cv2.CascadeClassifier('eye.xml')




st.set_page_config(
    page_title="🛑 Drowsiness Detection System",
    page_icon="😴",
    layout="wide",  # Expands to full width
    initial_sidebar_state="expanded"
)





# 🌑 Dark Theme & Centered Styling
st.markdown(
    """
    <style>
        body { background-color: #121212; color: #ffffff }
        .reportview-container { background-color: #121212; }
        .sidebar .sidebar-content { background-color: #1f1f1f; }
        h1, h2, h3, h4, h5, h6 { color: #ffffff; }
        .stButton>button { background-color: #ff4500; color: white; font-size: 18px; }
        .stTextInput>div>div>input { background-color: #222; color: white; font-size: 16px; }
        .stMarkdown p { font-size: 18px; line-height: 1.6; text-align: center; }
        .stWrite p { font-size: 18px; line-height: 1.6; text-align: left; }
        .stSelectbox>div>div>select { background-color: #222; color: white; font-size: 16px; }
    </style>
    """,
    unsafe_allow_html=True
)

# 🎨 Centered Title & Introduction
st.markdown("<h1 style='color: #FF1493; text-align: center;'>🛑 Drowsiness Detection System</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='color: #FFA500; text-align: center;'>👋 Welcome to the Drowsiness Detection System!</h2>", unsafe_allow_html=True)


st.write("🚀 Detect **drowsiness, yawning, and closed eyes** using ML Modell!")
# 🎯 Sidebar Input Selection


choice = st.sidebar.selectbox(
    "🎥 Select Input Type",
    ("🏠 Home", "📸 Image", "🎞️ Video", "📹 Device Camera", "🌐 Web IP Camera")
)

# 🏠 Home Screen
if choice == "🏠 Home":
    
    st.markdown(
    """
    - 🔹 **Select an input type** – Choose image, video, or live camera.
    - 📥 **Upload or Start Streaming** – Feed data into the system.
    - 🤖 **ML Detection Process** – Identifies yawning, closed eyes, and fatigue signs.
    """,
    unsafe_allow_html=True
)
    st.write("Select an input type from the sidebar to start the analysis.")
    st.success("🔍 This tool helps analyze drowsiness levels effectively to ensure safety and alertness!")




elif choice=='📸 Image':
    file=st.file_uploader('Upload an image')
    if file:
        b=file.getvalue()
        d=np.frombuffer(b, np.uint8)
        frame=cv2.imdecode(d,cv2.IMREAD_COLOR)
        face=facemodel.detectMultiScale(frame)
        for (x, y, l, w) in face:
            crop_face = frame[y:y+w, x:x+l]

            # Preprocess image directly
            crop_face = cv2.resize(crop_face, (150, 150))
            crop_face = img_to_array(crop_face)
            crop_face = np.expand_dims(crop_face, axis=0)

            # Predictions
            yawn_pred = yawnmodel.predict(crop_face)[0][0]
            drowsy_pred = drowsymodel.predict(crop_face)[0][0]
            eyes_pred = eyesmodel.predict(crop_face)[0][0]

            # Initialize detected conditions list
            detected_conditions = []
            color = (0, 255, 0)  # Default: Green for fresh face

            # Apply detections with colors and shapes
            if yawn_pred ==1:
                
                detected_conditions.append("Yawn Detected")
                color = (0, 0, 255)  # Red
                cv2.rectangle(frame, (x, y), (x+l, y+w), color, 2)  

            if drowsy_pred ==1:
                
                detected_conditions.append("Drowsy Detected")
                color = (255, 105, 180)  # Pink
                n=1.4
                cv2.ellipse(frame, (x + l//2, y + w//2), (int((w//3)*n), int((l//2)*n)), 0, 0, 360, (255, 105, 180), 2)  # Pink vertical oval for Drowsy Face
        

            if eyes_pred ==1:
                
                detected_conditions.append("Closed Eyes")
                color = (128, 0, 0)  # Maroon
                points = np.array([
                    [x, y], [x + l//2, y - w//4], [x + l, y],
                    [x + l, y + w], [x + l//2, y + w + w//4], [x, y + w]
                ], np.int32)
                cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)  

            if detected_conditions:
                pygame.mixer.music.load("beep.mp3")
                pygame.mixer.music.play()
            
            # If no detection, mark it as fresh face
            if not detected_conditions:
                detected_conditions.append("Fresh Face")
                cv2.rectangle(frame, (x, y), (x+l, y+w), (0, 255, 0), 2)  

            # Set starting position for text (Top-left corner)
            text_x, text_y = 10, 30  
            for condition in detected_conditions:
                cv2.putText(frame, condition, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                text_y += 30  # Move text down for each new condition
        st.image(frame, channels="BGR", width=400)

elif choice=='🎞️ Video':
    counter=0
    ecounter=0
    file=st.file_uploader('Upload a video')
    window=st.empty()
    if file:
        tfile=tempfile.NamedTemporaryFile()
        tfile.write(file.read())
        vid=cv2.VideoCapture(tfile.name)


        while (vid.isOpened()):
            flag, frame = vid.read()
            if flag: 
                face=facemodel.detectMultiScale(frame)
                for (x, y, l, w) in face:
                    crop_face = frame[y:y+w, x:x+l]

                    # Preprocess image directly
                    crop_face = cv2.resize(crop_face, (150, 150))
                    crop_face = img_to_array(crop_face)
                    crop_face = np.expand_dims(crop_face, axis=0)

                    # Predictions
                    yawn_pred = yawnmodel.predict(crop_face)[0][0]
                    drowsy_pred = drowsymodel.predict(crop_face)[0][0]
                    eyes_pred = eyesmodel.predict(crop_face)[0][0]

                    # Initialize detected conditions list
                    detected_conditions = []
                    color = (0, 255, 0)  # Default: Green for fresh face

                    # Apply detections with colors and shapes
                    if yawn_pred ==1:
                        counter=counter+1
                        detected_conditions.append("Yawn Detected")
                        color = (0, 0, 255)  # Red
                        cv2.rectangle(frame, (x, y), (x+l, y+w), color, 2)  

                    if drowsy_pred ==1:
                        counter=counter+1
                        detected_conditions.append("Drowsy Detected")
                        color = (255, 105, 180)  # Pink
                        n=1.4
                        cv2.ellipse(frame, (x + l//2, y + w//2), (int((w//3)*n), int((l//2)*n)), 0, 0, 360, (255, 105, 180), 2)  # Pink vertical oval for Drowsy Face
                

                    if eyes_pred ==1:
                        ecounter=ecounter+1
                        detected_conditions.append("Closed Eyes")
                        # pygame.mixer.music.load("beep.mp3")
                        # pygame.mixer.music.play()
                        color = (128, 0, 0)  # Maroon
                        points = np.array([
                            [x, y], [x + l//2, y - w//4], [x + l, y],
                            [x + l, y + w], [x + l//2, y + w + w//4], [x, y + w]
                        ], np.int32)
                        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)  

                    # If no detection, mark it as fresh face
                    if not detected_conditions:
                        detected_conditions.append("Fresh Face")
                        cv2.rectangle(frame, (x, y), (x+l, y+w), (0, 255, 0), 4)  

                    # Set starting position for text (Top-left corner)
                    text_x, text_y = 10, 30  
                    for condition in detected_conditions:
                        cv2.putText(frame, condition, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 1)
                        text_y += 30  # Move text down for each new condition
                window.image(frame,channels='BGR')

                if (counter %40==0) or (ecounter%10==0 and ecounter!=0):
                    pygame.mixer.music.load("beep.mp3")
                    pygame.mixer.music.play()

elif choice=='📹 Device Camera':
    counter=0
    ecounter=0

    window=st.empty()
    btn=st.button("Stop Camera")
    if btn:
        choice = 'Home'
        st.rerun()
    vid=cv2.VideoCapture(0)
    while (vid.isOpened()):
        flag, frame = vid.read()
        if flag: 
            face=facemodel.detectMultiScale(frame)
            for (x, y, l, w) in face:
                crop_face = frame[y:y+w, x:x+l]

                # Preprocess image directly
                crop_face = cv2.resize(crop_face, (150, 150))
                crop_face = img_to_array(crop_face)
                crop_face = np.expand_dims(crop_face, axis=0)

                # Predictions
                yawn_pred = yawnmodel.predict(crop_face)[0][0]
                drowsy_pred = drowsymodel.predict(crop_face)[0][0]
                eyes_pred = eyesmodel.predict(crop_face)[0][0]

                # Initialize detected conditions list
                detected_conditions = []
                color = (0, 255, 0)  # Default: Green for fresh face

                # Apply detections with colors and shapes
                if yawn_pred ==1:
                    counter=counter+1
                    detected_conditions.append("Yawn Detected")
                    color = (0, 0, 255)  # Red
                    cv2.rectangle(frame, (x, y), (x+l, y+w), color, 2)  

                if drowsy_pred ==1:
                    counter=counter+1
                    detected_conditions.append("Drowsy Detected")
                    color = (255, 105, 180)  # Pink
                    n=1.4
                    cv2.ellipse(frame, (x + l//2, y + w//2), (int((w//3)*n), int((l//2)*n)), 0, 0, 360, (255, 105, 180), 2)  # Pink vertical oval for Drowsy Face
            

                if eyes_pred ==1:
                    ecounter=ecounter+1
                    detected_conditions.append("Closed Eyes")
                    # pygame.mixer.music.load("beep.mp3")
                    # pygame.mixer.music.play()
                    color = (128, 0, 0)  # Maroon
                    points = np.array([
                        [x, y], [x + l//2, y - w//4], [x + l, y],
                        [x + l, y + w], [x + l//2, y + w + w//4], [x, y + w]
                    ], np.int32)
                    cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)  

                # If no detection, mark it as fresh face
                if not detected_conditions:
                    detected_conditions.append("Fresh Face")
                    cv2.rectangle(frame, (x, y), (x+l, y+w), (0, 255, 0), 4)  

                # Set starting position for text (Top-left corner)
                text_x, text_y = 10, 30  
                for condition in detected_conditions:
                    cv2.putText(frame, condition, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 1)
                    text_y += 30  # Move text down for each new condition
            window.image(frame,channels='BGR')

            if (counter %40==0) or (ecounter%10==0 and ecounter!=0):
                pygame.mixer.music.load("beep.mp3")
                pygame.mixer.music.play()

elif choice=='🌐 Web IP Camera':
    counter=0
    ecounter=0
    v = st.text_input("""Enter video URL (add '/video' in the end if needed)""")
    if not v.strip():
        st.warning("Please enter a valid video URL.")
        st.stop()
    window=st.empty()
    btn=st.button("Stop Camera")
    if btn:
        choice = 'Home'
        st.rerun()
    
    if v:
        vid = cv2.VideoCapture(v.strip())  # Removes extra spaces & captures the stream
        if not vid.isOpened():
            st.error("Unable to open video stream. Please check the URL.")
            vid.release()
            st.stop()

    while (vid.isOpened()):
        flag, frame = vid.read()
        if flag: 
            face=facemodel.detectMultiScale(frame)
            for (x, y, l, w) in face:
                crop_face = frame[y:y+w, x:x+l]

                # Preprocess image directly
                crop_face = cv2.resize(crop_face, (150, 150))
                crop_face = img_to_array(crop_face)
                crop_face = np.expand_dims(crop_face, axis=0)

                # Predictions
                yawn_pred = yawnmodel.predict(crop_face)[0][0]
                drowsy_pred = drowsymodel.predict(crop_face)[0][0]
                eyes_pred = eyesmodel.predict(crop_face)[0][0]

                # Initialize detected conditions list
                detected_conditions = []
                color = (0, 255, 0)  # Default: Green for fresh face

                # Apply detections with colors and shapes
                if yawn_pred ==1:
                    counter=counter+1
                    detected_conditions.append("Yawn Detected")
                    color = (0, 0, 255)  # Red
                    cv2.rectangle(frame, (x, y), (x+l, y+w), color, 2)  

                if drowsy_pred ==1:
                    counter=counter+1
                    detected_conditions.append("Drowsy Detected")
                    color = (255, 105, 180)  # Pink
                    n=1.4
                    cv2.ellipse(frame, (x + l//2, y + w//2), (int((w//3)*n), int((l//2)*n)), 0, 0, 360, (255, 105, 180), 2)  # Pink vertical oval for Drowsy Face
            

                if eyes_pred ==1:
                    ecounter=ecounter+1
                    detected_conditions.append("Closed Eyes")
                    # pygame.mixer.music.load("beep.mp3")
                    # pygame.mixer.music.play()
                    color = (128, 0, 0)  # Maroon
                    points = np.array([
                        [x, y], [x + l//2, y - w//4], [x + l, y],
                        [x + l, y + w], [x + l//2, y + w + w//4], [x, y + w]
                    ], np.int32)
                    cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)  

                # If no detection, mark it as fresh face
                if not detected_conditions:
                    detected_conditions.append("Fresh Face")
                    cv2.rectangle(frame, (x, y), (x+l, y+w), (0, 255, 0), 4)  

                # Set starting position for text (Top-left corner)
                text_x, text_y = 10, 30  
                for condition in detected_conditions:
                    cv2.putText(frame, condition, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 1)
                    text_y += 30  # Move text down for each new condition
            window.image(frame,channels='BGR')

            if (counter %40==0) or (ecounter%10==0 and ecounter!=0):
                pygame.mixer.music.load("beep.mp3")
                pygame.mixer.music.play()



st.write("---")
st.markdown(f"""**👨‍💻 Developed by Ashish Bonde** <br> 
💬 **Interested in the Drowsiness Detection System?** <br> 
📲 Connect with me on :<br>[LinkedIn](https://www.linkedin.com/in/ashish-bonde/)<br>[GitHub Profile](https://github.com/Ashish-Bonde)<br>
[WhatsApp](https://api.whatsapp.com/send?phone=918484864084&text=Hi%20Ashish!%20I%20came%20across%20your%20Drowsiness%20Detection%20System%20and%20would%20love%20to%20connect%20to%20learn%20more%20about%20it.%20Let's%20connect!)
""", unsafe_allow_html=True)




