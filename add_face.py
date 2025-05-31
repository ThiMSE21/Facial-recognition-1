import cv2
import face_recognition
import requests
import json
import numpy as np

cap = cv2.VideoCapture(0)
print("[INFO] Mở webcam, nhấn 'c' để chụp khuôn mặt, 'q' để thoát")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1)

    if key == ord("c"):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if face_encodings:
            name = input("Nhập tên cho khuôn mặt này: ")
            encoding_list = face_encodings[0].tolist()
            data = {"name": name, "encoding": encoding_list}
            try:
                res = requests.post("http://127.0.0.1:5000/add_face", json=data)
                print("[INFO]", res.json())
            except Exception as e:
                print("[ERROR]", e)
            print("[SUCCESS] Đã lưu khuôn mặt của", name)
        else:
            print("[WARN] Không tìm thấy khuôn mặt!")
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
