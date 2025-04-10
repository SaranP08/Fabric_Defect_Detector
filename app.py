import streamlit as st
import numpy as np
import cv2
from PIL import Image
from detector import detect_damage
import tempfile
import os
import zipfile

st.set_page_config(page_title="🧠 Multi-Source Defect Detection", layout="wide")
st.title("🧵 Fabric Defect Detection Dashboard")
st.write("Choose an input source and detect Tears, Holes, Unstitched areas using YOLOv8")

# --- Input mode selector ---
input_mode = st.radio("Select Input Source", ["Image", "Video", "Webcam", "Image Folder"])

# --- Image Upload ---
if input_mode == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, caption="Original Image", use_column_width=True)

        annotated, results = detect_damage(image)
        st.image(annotated[:, :, ::-1], caption="Detected Defects", use_column_width=True)
        st.write("Detection Results:", results)

# --- Video Upload ---
elif input_mode == "Video":
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated, _ = detect_damage(frame)
            stframe.image(annotated[:, :, ::-1], channels="RGB")
        cap.release()
        st.success("🎞️ Video processed.")

# --- Webcam Live Detection ---
elif input_mode == "Webcam":
    st.write("🔴 Starting webcam... press 'Stop' to exit")
    run = st.checkbox("Start Camera")
    frame_display = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("🚫 Failed to read from webcam")
                break
            annotated, _ = detect_damage(frame)
            frame_display.image(annotated[:, :, ::-1], channels="RGB")
        cap.release()

# --- Folder of Images (Zip Upload) ---
elif input_mode == "Image Folder":
    zip_file = st.file_uploader("Upload ZIP of images", type=["zip"])
    if zip_file:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(tmpdir)
            image_paths = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            
            for img_path in image_paths:
                image = cv2.imread(img_path)
                if image is not None:
                    annotated, results = detect_damage(image)
                    st.image(annotated[:, :, ::-1], caption=f"🖼️ {os.path.basename(img_path)}", use_column_width=True)
                    st.write(results)
