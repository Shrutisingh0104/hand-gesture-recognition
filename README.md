# Hand Gesture Recognition using Machine Learning & Computer Vision

##  Overview
This project is a real-time Hand Gesture Recognition System that uses Computer Vision and Machine Learning to detect and classify hand gestures through a webcam. It processes live video input and identifies gestures accurately using trained models and hand tracking techniques.

---

##  Features
- Real-time hand detection using webcam  
- Accurate gesture recognition  
- Smooth and fast processing  
- Works in live video stream  
- User-friendly and lightweight system  

---

##  Tech Stack
- **Python**
- **OpenCV (cv2)** – Image processing and webcam handling  
- **MediaPipe** – Hand tracking and landmark detection  
- **TensorFlow** – Machine Learning model (gesture classification)  
- **NumPy** – Numerical computations  

---

##  How It Works
1. Captures live video from webcam  
2. Detects hand landmarks using MediaPipe  
3. Extracts key points from hand gestures  
4. Sends data to trained ML model (TensorFlow)  
5. Predicts and displays the gesture in real time  

---

##  Installation


# 1. Clone the repository
git clone https://github.com/Shrutisingh0104/hand-gesture-recognition.git

# 2. Navigate to project folder
cd hand-gesture-recognition

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the project
python app.py
│
├── app.py                  # Main application file
├── model/                 # Trained ML model 
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── dataset/ (optional)    # Training data
 Output
<img width="150" height="100" alt="Screenshot 2026-05-30 132031" src="https://github.com/user-attachments/assets/58d9225b-3f50-4abe-b119-0fa8c3e7bf9b" />
<img width="150" height="100" alt="Screenshot 2026-05-30 132031" src="https://github.com/user-attachments/assets/e7029e4d-8915-44e8-9b84-878ab090ea0d" />
<img width="150" height="100" alt="Screenshot 2026-05-30 132149" src="https://github.com/user-attachments/assets/894097a6-6fab-420c-a59b-b9caffbee8aa" />




