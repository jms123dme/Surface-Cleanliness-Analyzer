import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# Create a folder for uploads
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to analyze surface cleanliness
def analyze_surface_cleanliness(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Image not found or unable to read the image.", 0, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
    edge_density = np.sum(edges > 0) / edges.size * 100
    status = "Clean" if edge_density < 2.0 else "Dirty"
    return status, edge_density, img

# Streamlit UI
st.title("Surface Cleanliness Analyzer")

uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
if uploaded_files:
    st.write("Analyzing uploaded images...")
    image_data = []

    for uploaded_file in uploaded_files:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Analyze image
        status, edge_density, img = analyze_surface_cleanliness(file_path)
        if img is not None:
            image_data.append((uploaded_file.name, status, edge_density, img))

    # Display results
    st.write("### Analysis Results:")
    for file_name, status, edge_density, img in image_data:
        col1, col2 = st.columns([1, 3])

        # Show image
        with col1:
            st.image(img, caption=f"{file_name}", use_column_width=True, channels="BGR")

        # Show analysis results
        with col2:
            st.write(f"**File Name:** {file_name}")
            st.write(f"**Surface Condition:** {status}")
            st.write(f"**Edge Density (%):** {edge_density:.2f}")
            conclusion = "Well-maintained" if status == "Clean" else "Requires cleaning"
            st.write(f"**Conclusion:** {conclusion}")
