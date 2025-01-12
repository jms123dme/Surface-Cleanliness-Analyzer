import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import textwrap
from skimage.feature import local_binary_pattern


# Set up upload folder dynamically
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to analyze surface cleanliness (basic and advanced)
def analyze_surface_cleanliness(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # Apply edge detection using Canny
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
    
    # Calculate the percentage of detected edges
    edge_density = np.sum(edges > 0) / edges.size * 100
    
    # Texture analysis using Local Binary Pattern (LBP)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= lbp_hist.sum()  # Normalize
    
    # Spot/blob detection
    detector = cv2.SimpleBlobDetector_create()
    keypoints = detector.detect(blurred)
    blob_count = len(keypoints)
    
    # Decision criteria
    status = "Clean" if edge_density < 2.0 and blob_count < 10 else "Dirty"
    
    # Detailed analysis summary
    analysis_details = {
        "Edge Density (%)": edge_density,
        "Blob Count": blob_count,
        "LBP Uniformity": lbp_hist[1],
    }
    return status, analysis_details, edges

# Function to generate a summary table with full borders and detailed analysis
def generate_summary_table(image_data, clean_count, dirty_count, cleanliness_percentage, save_path=None):
    fig, ax = plt.subplots(figsize=(16, len(image_data) * 3 + 1))  # Adjust height dynamically based on rows
    ax.axis("off")
    
    # Table headers
    headers = ["Image", "File Name", "Surface Condition", "Edge Density (%)", "Blob Count", "LBP Uniformity", "Conclusion"]
    n_cols = len(headers)
    n_rows = len(image_data) + 2  # Include 1 extra row for scoring

    # Wrap text function
    def wrap_text(text, width):
        return "\n".join(textwrap.wrap(text, width))

    # Add headers to the table
    for col, header in enumerate(headers):
        ax.text(
            (col + 0.5) / n_cols, 1 - 0.5 / n_rows, header, weight="bold", fontsize=12, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#40466e", edgecolor="black"), color="white"
        )
    
    # Add image data to the table
    for row, (file_name, status, analysis_details, img) in enumerate(image_data):
        # Row borders
        ax.plot([0, 1], [1 - ((row + 1) / n_rows), 1 - ((row + 1) / n_rows)], color="black", lw=1)  # Horizontal lines
        for col in range(1, n_cols):
            ax.plot([col / n_cols, col / n_cols], [0, 1], color="black", lw=1)  # Vertical lines

        # Add images and text to the table cells
        thumbnail = cv2.resize(img, (80, 80))  # Ensure the image fits in the cell
        image_box = OffsetImage(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB), zoom=1)
        ab = AnnotationBbox(image_box, ((0.5 / n_cols), 1 - ((row + 1 + 0.5) / n_rows)), frameon=False)
        ax.add_artist(ab)

        # File name (wrapped text)
        ax.text((1 + 0.5) / n_cols, 1 - ((row + 1 + 0.5) / n_rows),
                wrap_text(file_name, 25), fontsize=10, ha="center", va="center")
        
        # Surface condition
        ax.text((2 + 0.5) / n_cols, 1 - ((row + 1 + 0.5) / n_rows),
                wrap_text(status, 15), fontsize=10, ha="center", va="center")
        
        # Edge density
        ax.text((3 + 0.5) / n_cols, 1 - ((row + 1 + 0.5) / n_rows),
                wrap_text(f"{analysis_details['Edge Density (%)']:.2f}", 10), fontsize=10, ha="center", va="center")
        
        # Blob count
        ax.text((4 + 0.5) / n_cols, 1 - ((row + 1 + 0.5) / n_rows),
                wrap_text(str(analysis_details['Blob Count']), 10), fontsize=10, ha="center", va="center")
        
        # LBP Uniformity
        ax.text((5 + 0.5) / n_cols, 1 - ((row + 1 + 0.5) / n_rows),
                wrap_text(f"{analysis_details['LBP Uniformity']:.4f}", 10), fontsize=10, ha="center", va="center")
        
        # Conclusion
        conclusion = "Well-maintained" if status == "Clean" else "Requires cleaning"
        ax.text((6 + 0.5) / n_cols, 1 - ((row + 1 + 0.5) / n_rows),
                wrap_text(conclusion, 15), fontsize=10, ha="center", va="center")
    
    # Add scoring row at the bottom
    scoring_text = f"Total Images: {len(image_data)} | Clean Images: {clean_count} | Dirty Images: {dirty_count} | Cleanliness Percentage: {cleanliness_percentage:.2f}%"
    ax.text(0.5, 0.5 / n_rows, wrap_text(scoring_text, 60), fontsize=12, ha="center", va="center", color="black", 
            bbox=dict(boxstyle="round,pad=0.5", edgecolor="black", facecolor="lightgray"))

    # Add full outer border
    ax.plot([0, 1], [1, 1], color="black", lw=2)  # Top border
    ax.plot([0, 1], [0, 0], color="black", lw=2)  # Bottom border
    ax.plot([0, 0], [0, 1], color="black", lw=2)  # Left border
    ax.plot([1, 1], [0, 1], color="black", lw=2)  # Right border
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig

# Streamlit App
st.title("Advanced Surface Cleanliness Analyzer")

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
        status, analysis_details, edges = analyze_surface_cleanliness(image)
        image_data.append((uploaded_file.name, status, analysis_details, image))

    # Calculate scoring
    total_images = len(image_data)
    clean_images = sum(1 for _, status, _, _ in image_data if status == "Clean")
    dirty_images = total_images - clean_images
    cleanliness_percentage = (clean_images / total_images) * 100 if total_images > 0 else 0

    # Generate and display the summary table
    st.write("### Summary Table:")
    summary_path = os.path.join(UPLOAD_FOLDER, "summary_table.png")
    summary_fig = generate_summary_table(image_data, clean_images, dirty_images, cleanliness_percentage, save_path=summary_path)
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

    # Display scoring in Streamlit
    st.write("### Scoring:")
    st.write(f"- **Total Images:** {total_images}")
    st.write(f"- **Clean Images:** {clean_images}")
    st.write(f"- **Dirty Images:** {dirty_images}")
    st.write(f"- **Cleanliness Percentage:** {cleanliness_percentage:.2f}%")
