Thank you for pointing that out. I'll create a **clean, well-formatted `README.md` file** for your GitHub repository based on the **Drowsiness Detection System** code and project structure.

---

## ✅ Here's a Properly Formatted `README.md` File

```markdown
# 🚨 Drowsiness Detection System  
## Real-Time Yawn, Closed Eyes & Fatigue Detection Using ML

---

### 🔗 Live WebApp (Deployed on Cloud):  
👉 [https://drowsiness-detection-system-webapp.streamlit.app/](https://drowsiness-detection-system-webapp.streamlit.app/)

### 🌐 GitHub Repository:  
👉 [https://github.com/Ashish-Bonde/Drowsiness-Detection-System](https://github.com/Ashish-Bonde/Drowsiness-Detection-System)

---

## 📌 Overview

The **Drowsiness Detection System** is a real-time web application designed to detect signs of drowsiness, yawning, and closed eyes using **Machine Learning models**. This system leverages **OpenCV**, **TensorFlow/Keras**, and **Streamlit** to provide an interactive and user-friendly experience.

This project is especially useful in safety-critical environments like **driver monitoring systems**, **workplace alertness checks**, or **healthcare fatigue detection**.

---

## ✅ Key Features

| Feature | Description |
|--------|-------------|
| 📸 **Image Input** | Upload an image and analyze for drowsy indicators |
| 🎥 **Video Analysis** | Analyze pre-recorded videos for continuous monitoring |
| 📹 **Device Camera** | Use your webcam for live drowsiness detection |
| 🌐 **IP Camera Stream** | Connect to IP cameras for remote surveillance |
| 🔊 **Audio Alerts** | Beep sound alerts when drowsiness/yawning is detected |
| 🧠 **ML-Powered Detection** | Uses trained models for accurate prediction of drowsiness |

---

## 🛠️ Technology Stack

| Layer | Technology Used |
|-------|------------------|
| **Frontend** | Streamlit |
| **Backend** | Python |
| **Model Framework** | TensorFlow / Keras |
| **Computer Vision** | OpenCV |
| **Audio Alert** | Pygame |
| **UI Styling** | CSS + HTML (Embedded in Streamlit) |

---

## 🧩 Models Used

This system uses **pre-trained CNN models** for the following tasks:

| Model File | Purpose |
|-----------|---------|
| `eye.h5` | Detects whether eyes are open or closed |
| `yawn.h5` | Detects yawning behavior |
| `drowsy.h5` | Identifies overall drowsiness based on facial features |

> These models are trained using datasets containing images of faces with various expressions and states.

---

## 📁 Project Structure

```
Drowsiness-Detection-System/
│
├── app.py                      # Main application file
├── models/                     # Folder containing trained models
│   ├── eye.h5                  # Eye closure detection model
│   ├── yawn.h5                 # Yawning detection model
│   └── drowsy.h5               # Drowsiness classification model
├── haarcascades/               # Haar Cascade classifiers
│   ├── face.xml                # Face detection classifier
│   └── eye.xml                 # Eye detection classifier
├── beep.mp3                    # Audio alert sound
└── requirements.txt            # List of required packages
```

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

### `requirements.txt` Includes:
```
streamlit==1.43.2
opencv-python==4.8.0
tensorflow==2.13.0
keras==2.17.0
pygame==2.5.2
Pillow==10.0.1
scipy==1.11.3
numpy==1.26.1
```

---

## ▶️ How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ashish-Bonde/Drowsiness-Detection-System.git
   cd Drowsiness-Detection-System
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. **Access the app**
   Open your browser and go to:
   ```
   http://localhost:8501
   ```

---

## 🎯 Supported Input Sources

You can select from the following input sources in the sidebar:

1. **📸 Image**: Upload and analyze a static image.
2. **🎞️ Video**: Upload and process a video file.
3. **📹 Device Camera**: Use your laptop/desktop webcam for live analysis.
4. **🌐 Web IP Camera**: Enter an RTSP/IP camera URL for remote monitoring.

---

## 📊 Detection Visualization

When any condition is detected, the system overlays:
- **Red Rectangle** – Yawn Detected
- **Pink Oval** – Drowsy Detected
- **Maroon Polygon** – Closed Eyes Detected
- **Green Rectangle** – Fresh Face (No drowsiness detected)

It also plays a **beep sound** as an alert when critical conditions are met.

---

## 🚨 Limitations

| Limitation | Description |
|----------|-------------|
| 🌑 Lighting Conditions | Performance may degrade in low-light or backlit environments |
| 🔄 Pose Sensitivity | Best results when the subject faces the camera directly |
| ⚙️ Hardware Dependent | Real-time performance depends on CPU/GPU capabilities |
| 📶 Network Latency | IP camera feed may lag depending on network speed |

---

## 🚀 Future Enhancements

- 🧠 **Improve Accuracy**: Retrain models with more diverse datasets
- 📱 **Mobile App Integration**: Convert to mobile apps using TensorFlow Lite
- 🖥️ **Desktop App**: Build standalone applications using PyQt or Tkinter
- 📡 **IoT Integration**: Integrate with wearable devices or vehicle sensors
- 📊 **Dashboard Analytics**: Add dashboard for tracking alert frequency and patterns

---

## 👨‍💻 Developer Info

- **Name:** Ashish Bonde  
- **GitHub:** [github.com/Ashish-Bonde](https://github.com/Ashish-Bonde)  
- **LinkedIn:** [linkedin.com/in/ashish-bonde](https://www.linkedin.com/in/ashish-bonde/)  
- **WhatsApp:** [+91 8484864084](https://api.whatsapp.com/send?phone=918484864084&text=Hi%20Ashish!%20I'm%20interested%20in%20your%20Drowsiness%20Detection%20System.%20Let's%20connect!)

---

## 📜 License

MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🙌 Contributing

Contributions are welcome! Feel free to open issues or pull requests for improvements, bug fixes, or new features.

---

Thank you for checking out the **Drowsiness Detection System**!  
We hope it helps enhance safety and awareness in your applications. 😊
```

