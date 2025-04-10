import streamlit as st
import numpy as np
import cv2
from PIL import Image
from detector import detect_damage

st.set_page_config(page_title="🧵 Fabric Defect Detector", layout="centered")
st.title("🧠 Fabric Defect Detection with YOLOv8")
st.write("Upload an image and we'll detect **Tears**, **Unstitched areas**, and **Holes** using a trained YOLOv8 model.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="🖼️ Original Image", use_column_width=True)

    # Inference
    st.write("🔍 Detecting defects...")
    annotated_img, results = detect_damage(image)

    # Display annotated image
    st.image(annotated_img[:, :, ::-1], caption="✅ Detected Defects", use_column_width=True)

    # Display results
    if results:
        st.subheader("📋 Defect Report")
        for i, res in enumerate(results):
            st.markdown(f"**{i+1}. {res['label']}**")
            st.write(f"Confidence: `{res['confidence']}`")
            st.write(f"BBox: `{res['bbox']}`")
            st.write(f"Center: `{res['center']}`")
    else:
        st.success("🎉 No defects found!")

