from flask import Flask, request, jsonify
import face_recognition
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Load known encodings
known_encodings = []
known_names = []

for file in os.listdir("known_faces"):
    if file.endswith(".npy"):
        name = file[:-4]
        encoding = np.load(f"known_faces/{file}")
        known_encodings.append(encoding)
        known_names.append(name)

@app.route("/recognize", methods=["POST"])
def recognize():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    image = Image.open(image_file).convert("RGB")
    image_np = np.array(image).astype(np.uint8)

    face_locations = face_recognition.face_locations(image_np)
    face_encodings = face_recognition.face_encodings(image_np, face_locations)

    results = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            matched_idx = matches.index(True)
            name = known_names[matched_idx]

        results.append(name)

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
