import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
import sys
import time
warnings.filterwarnings("ignore")
from keras.models import load_model

# Set default encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Load the model
model = load_model(r"Model Path")
print("Model loaded successfully")

# Load the Haar Cascade for face detection
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("Haar Cascade loaded successfully")

# Start video capture
cap = cv2.VideoCapture(0)
print("Video capture started")

while True:
    ret, test_img = cap.read()  # Capture frame
    if not ret:
        print("Failed to capture image")
        continue
    
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    print("Image converted to grayscale")

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"Faces detected: {len(faces_detected)}")

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + h, x:x + w]  # Crop the face area
        roi_gray = cv2.resize(roi_gray, (48, 48))  # Resize to 48x48
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0

        try:
            predictions = model.predict(img_pixels)
            print(f"Predictions: {predictions}")
        except UnicodeEncodeError as e:
            print(f"UnicodeEncodeError: {e}")
            continue  # Skip this iteration and continue with the next frame

        # Find max indexed array
        max_index = np.argmax(predictions[0])
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis', resized_img)

    # Use cv2.waitKey() to check for user input and add delay
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key to exit
        print("ESC key pressed. Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
