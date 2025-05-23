import cv2
import face_recognition
import numpy as np
from PIL import Image
import os

# Create folder if it doesn't exist
if not os.path.exists("known_faces"):
    os.makedirs("known_faces")

print("[INFO] Press 's' to capture and save the image")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Add Face", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        name = input("Enter person's name: ")
        file_path = f"known_faces/{name}.jpg"
        cv2.imwrite(file_path, frame)
        print(f"[OK] Image saved: {file_path}")
        break

cap.release()
cv2.destroyAllWindows()

# Load and ensure the image is in RGB and uint8 format
image = Image.open(file_path).convert("RGB")
image = np.array(image).astype(np.uint8)

# Encode the face
encodings = face_recognition.face_encodings(image)

if len(encodings) == 0:
    print("[ERROR] No face detected in the image!")
else:
    np.save(f"known_faces/{name}.npy", encodings[0])
    print(f"[INFO] Encoding saved: known_faces/{name}.npy")
