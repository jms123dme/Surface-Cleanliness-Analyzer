import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Set up upload folder dynamically
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to analyze surface cleanliness
def analyze_surface_cleanliness(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # Apply edge detection using Canny
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
    
    # Calculate the percentage of detected edges
    edge_density = np.sum(edges > 0) / edges.size * 100
    
    # Threshold for cleanliness
    status = "Clean" if edge_density < 2.0 else "Dirty"
    return status, edge_density, edges

# Function to generate a summary table with borders and alignment
def generate_summary_table(image_data, save_path=None):
    fig, ax = plt.subplots(figsize=(12, len(image_data) * 2.5))
    ax.axis("off")
    
    # Table headers
    headers = ["Image", "File Name", "Surface Condition", "Edge Density (%)", "Conclusion"]
    n_cols = len(headers)
    n_rows = len(image_data) + 1

    # Add headers to the table
    for col, header in enumerate(headers):
        ax.text(
            (col + 0.5) / n_cols, 1 - 0.5 / n_rows, header, weight="bold", fontsize=12, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#40466e", edgecolor="black"), color="white"
        )
    
    # Add image data to the table
    for row, (file_name, status, edge_density, img) in enumerate(image_data):
        # Draw borders
        ax.plot([0, 1], [1 - ((row + 1) / n_rows), 1 - ((row + 1) / n_rows)], color="black", lw=1)  # Horizontal lines
        for col in range(1, n_cols):
            ax.plot([col / n_cols, col / n_cols], [0, 1], color="black", lw=1)  # Vertical lines
        
        thumbnail = cv2.resize(img, (50, 50))
        image_box = OffsetImage(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB), zoom=0.8)
        ab = AnnotationBbox(image_box, (0.1, 1 - ((row + 1 + 0.5) / n_rows)), frameon=False)
        ax.add_artist(ab)

        # File name
        ax.text((1 + 0.5) / n_cols, 1 - ((row + 1 + 0.5) / n_rows), file_name, fontsize=10, ha="center", va="center")
        # Surface condition
        ax.text((2 + 0.5) / n_cols, 1 - ((row + 1 + 0.5) / n_rows), status, fontsize=10, ha="center", va="center")
        # Edge density
        ax.text((3 + 0.5) / n_cols, 1 - ((row + 1 + 0.5) / n_rows), f"{edge_density:.2f}%", fontsize=10, ha="center", va="center")
        # Conclusion
        conclusion = "Well-maintained" if status == "Clean" else "Requires cleaning"
        ax.text((4 + 0.5) / n_cols, 1 - ((row + 1 + 0.5) / n_rows), conclusion, fontsize=10, ha="center", va="center")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig

# Streamlit App
st.title("Surface Cleanliness Analyzer")

# File upload section
uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded_files:
    st.write("Analyzing uploaded images...")
    image_data = []

    for uploaded_file in uploaded_files:
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Read the image with OpenCV
        image = cv2.imread(file_path)
        if image is None:
            st.warning(f"Could not read file: {uploaded_file.name}")
            continue

        # Analyze image
        status, edge_density, edges = analyze_surface_cleanliness(image)
        image_data.append((uploaded_file.name, status, edge_density, image))

    # Display results
    st.write("### Analysis Results:")
    for file_name, status, edge_density, img in image_data:
        col1, col2 = st.columns([1, 3])

        # Show the uploaded image
        with col1:
            st.image(img, caption=file_name, use_column_width=True, channels="BGR")

        # Show analysis details
        with col2:
            st.write(f"**File Name:** {file_name}")
            st.write(f"**Surface Condition:** {status}")
            st.write(f"**Edge Density (%):** {edge_density:.2f}")
            conclusion = "Well-maintained" if status == "Clean" else "Requires cleaning"
            st.write(f"**Conclusion:** {conclusion}")

    # Generate and display the summary table
    st.write("### Summary Table:")
    summary_path = os.path.join(UPLOAD_FOLDER, "summary_table.png")
    summary_fig = generate_summary_table(image_data, save_path=summary_path)
    st.pyplot(summary_fig)

    # Add download button for the summary table
    with open(summary_path, "rb") as f:
        summary_data = f.read()
    st.download_button(
        label="Download Summary Table as PNG",
        data=summary_data,
        file_name="summary_table.png",
        mime="image/png"
    )
