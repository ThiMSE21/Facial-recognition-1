from flask import Flask, request, jsonify
import os
import pickle

app = Flask(__name__)
FACE_DATA_FILE = "known_faces.pkl"

# Load dữ liệu nếu có
if os.path.exists(FACE_DATA_FILE):
    with open(FACE_DATA_FILE, "rb") as f:
        known_faces = pickle.load(f)
else:
    known_faces = {}

@app.route("/add_face", methods=["POST"])
def add_face():
    data = request.get_json()
    name = data["name"]
    encoding = data["encoding"]

    if name in known_faces:
        known_faces[name].append(encoding)
    else:
        known_faces[name] = [encoding]

    with open(FACE_DATA_FILE, "wb") as f:
        pickle.dump(known_faces, f)

    return jsonify({"message": f"Đã thêm khuôn mặt cho {name}"}), 200

@app.route("/known_faces", methods=["GET"])
def get_known_faces():
    return jsonify(known_faces), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
