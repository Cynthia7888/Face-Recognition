

import streamlit as st
import cv2
import numpy as np
import io
import time
from PIL import Image
import tensorflow as tf
import os
import gdown

# === Load model ===
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model("resnet.h5")
# import os, gdown, tensorflow as tf, streamlit as st

FILE_ID = "1xPzhxtvcbuoSXuFqPTqD-UYOmp00gnn3"
URL = f"https://drive.google.com/uc?id={FILE_ID}"
LOCAL = "resnet.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(LOCAL) or os.path.getsize(LOCAL) < 1024:
        gdown.download(URL, LOCAL, quiet=False)
    return tf.keras.models.load_model(LOCAL)

model = load_model()

# === Class labels (from your train_ds.class_indices) ===
inv_class_names = {
    0: "Than_Thar", 1: "Su_dad", 2: "nandar", 3: "mayzin",
    4: "mom", 5: "pyonemoh", 6: "Lwin", 7: "Su"
}

# === Face Detection Functions (unchanged) ===
def detect_and_crop_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=3)
    boxes = []
    for (x, y, w, h) in faces:
        boxes.append((x, y, x + w, y + h))
    return boxes

def detect_and_crop_faces_dnn(image, conf_threshold=0.5):
    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt"

    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()

    cropped_faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            # clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            if x2 > x1 and y2 > y1:
                cropped_faces.append((x1, y1, x2, y2))
    return cropped_faces

# === Predict Face Identity (unchanged) ===
def predict_face_identity(face_region):
    face = cv2.resize(face_region, (512, 512))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face / 255.0
    face = np.expand_dims(face, axis=0)
    pred = model.predict(face)
    class_idx = np.argmax(pred, axis=1)[0]
    return inv_class_names.get(class_idx, "Unknown")

# === Streamlit UI ===
st.title("Face Detection and Recognition")

detection_method = st.radio("Choose detection method", ("Haar Cascade", "DNN"))
input_option = st.radio("Image source:", ("Browse Image", "Capture Image"))

# === Helper to process image ===
def process_faces(image_np, boxes):
    result_image = image_np.copy()
    predictions = []

    for (x1, y1, x2, y2) in boxes:
        face = image_np[y1:y2, x1:x2]
        if face is not None and face.size > 0:
            label = predict_face_identity(face)
            predictions.append(label)
            # Draw box and label
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result_image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    return result_image, predictions

# === Upload Image ===
if input_option == "Browse Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")

        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if detection_method == "Haar Cascade":
            boxes = detect_and_crop_faces(image_np)
        else:
            boxes = detect_and_crop_faces_dnn(image_np)

        result_img, preds = process_faces(image_np, boxes)
        st.image(result_img, caption="Detected Faces")
        st.write(f"Detected {len(preds)} face(s)")
        for i, label in enumerate(preds):
            st.write(f"Face {i+1}: {label}")

# === Capture Image ===
elif input_option == "Capture Image":
    captured = st.camera_input("Take a photo")
    if captured:
        image = Image.open(io.BytesIO(captured.getvalue()))
        st.image(image, caption="Captured Image")

        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if detection_method == "Haar Cascade":
            boxes = detect_and_crop_faces(image_np)
        else:
            boxes = detect_and_crop_faces_dnn(image_np)

        result_img, preds = process_faces(image_np, boxes)
        st.image(result_img, caption="Detected Faces")
        st.write(f"Detected {len(preds)} face(s)")
        for i, label in enumerate(preds):
            st.write(f"Face {i+1}: {label}")
