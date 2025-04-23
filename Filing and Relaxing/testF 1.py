import cv2
import numpy as np
from keras.models import load_model
import pygame
import tkinter as tk
from PIL import Image, ImageTk
from collections import deque, Counter
import time

# Load the emotion detection model
model = load_model('emotion_model.hdf5', compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Settings for each emotion: background color, music, and scent suggestion
emotion_effects = {
    'Angry': {
        'color': '#98FF98',
        'music': 'music/calm_sea_waves.mp3',
        'scent': 'Sandalwood'
    },
    'Disgust': {
        'color': '#A9BA9D',
        'music': 'music/small_river_flow.mp3',
        'scent': 'Mint'
    },
    'Fear': {
        'color': '#E6E6FA',
        'music': 'music/The_sound_of_a_small_fire.mp3',
        'scent': 'Vanilla'
    },
    'Happy': {
        'color': '#FFD700',
        'music': 'music/garden.mp3',
        'scent': 'Jasmine'
    },
    'Sad': {
        'color': '#ADD8E6',
        'music': 'music/swaying_leaves.mp3',
        'scent': 'Frankincense'
    },
    'Surprise': {
        'color': '#F5F5DC',
        'music': 'music/deep_breathing.mp3',
        'scent': 'Lavender with Citrus'
    },
    'Neutral': {
        'color': '#D3D3D3',
        'music': 'music/rustling_of_trees.mp3',
        'scent': 'Green Tea'
    }
}

# Initialize music player
pygame.mixer.init()
current_music = None

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to play music based on emotion
def play_music(emotion):
    global current_music
    music_path = emotion_effects[emotion]['music']
    if current_music != music_path:
        pygame.mixer.music.fadeout(500)
        pygame.mixer.music.load(music_path)
        pygame.mixer.music.play(-1)
        current_music = music_path

# GUI setup
root = tk.Tk()
root.title("Emotion Detection & Relaxation")
root.geometry("800x600")

# Camera feed display
label = tk.Label(root)
label.pack()

# Emotion label
emotion_text = tk.Label(root, text="", font=("Arial", 24))
emotion_text.pack(pady=20)

# Scent suggestion label
scent_text = tk.Label(root, text="", font=("Arial", 18))
scent_text.pack()

# Start camera
cap = cv2.VideoCapture(0)

# Emotion queue to determine stable emotion
emotion_queue = deque(maxlen=15)
stable_emotion = "Neutral"
emotion_start_time = time.time()

def update_frame():
    global stable_emotion, emotion_start_time

    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    detected_emotion = "Neutral"

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64))
        roi = np.expand_dims(roi_gray, axis=(0, -1)) / 255.0
        prediction = model.predict(roi, verbose=0)
        emotion = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction)
        detected_emotion = emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"{emotion} ({confidence*100:.1f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Update emotion queue
    emotion_queue.append(detected_emotion)
    most_common_emotion = Counter(emotion_queue).most_common(1)[0][0]

    # Change music and background only if emotion changed
    if most_common_emotion != stable_emotion:
        stable_emotion = most_common_emotion
        emotion_start_time = time.time()
        play_music(stable_emotion)
        root.config(bg=emotion_effects[stable_emotion]['color'])

    # Update labels with current emotion and duration
    duration = int(time.time() - emotion_start_time)
    emotion_text.config(text=f"Emotion: {stable_emotion} ({duration} sec)")

    # Show recommended scent
    scent = emotion_effects[stable_emotion]['scent']
    scent_text.config(text=f"Scent: {scent}")

    # Display camera frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)

    root.after(10, update_frame)

# Cleanup on exit
def on_closing():
    cap.release()
    pygame.mixer.music.stop()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
update_frame()
root.mainloop()
