import cv2
import face_recognition
import requests
import numpy as np

print("[INFO] Lấy dữ liệu khuôn mặt đã biết từ server...")
try:
    response = requests.get("http://127.0.0.1:5000/known_faces")
    known_faces = response.json()
except Exception as e:
    print("[ERROR] Không thể lấy dữ liệu từ server:", e)
    known_faces = {}

known_encodings = []
known_names = []

for name, encodings in known_faces.items():
    for enc in encodings:
        known_encodings.append(np.array(enc))
        known_names.append(name)

cap = cv2.VideoCapture(0)
print("[INFO] Mở webcam, nhấn 'q' để thoát")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    cv2.imshow("Nhận diện khuôn mặt", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
