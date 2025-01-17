import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import textwrap

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
    status = "Clean" if edge_density < 3.0 else "Dirty"  # Updated threshold to <3%
    return status, edge_density, edges

# Function to generate a summary table with full borders and scoring
def generate_summary_table(image_data, clean_count, dirty_count, cleanliness_percentage, save_path=None):
    fig, ax = plt.subplots(figsize=(12, len(image_data) * 2.5 + 1))  # Extra space for scoring row
    ax.axis("off")
    
    # Table headers
    headers = ["Image", "File Name", "Surface Condition", "Edge Density (%)", "Conclusion"]
    n_cols = len(headers)
    n_rows = len(image_data) + 2  # Include 1 extra row for scoring

    # Wrap text function
    def wrap_text(text, width):
        return "\n".join(textwrap.wrap(text, width))

    # Add headers to the table
    for col, header in enumerate(headers):
        ax.text(
            (col + 0.5) / n_cols, 1 - 0.5 / n_rows, header, weight="bold", fontsize=14, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#40466e", edgecolor="black"), color="white"
        )
    
    # Add image data to the table
    for row, (file_name, status, edge_density, img) in enumerate(image_data):
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
                wrap_text(file_name, 20), fontsize=12, ha="center", va="center")
        
        # Surface condition
        ax.text((2 + 0.5) / n_cols, 1 - ((row + 1 + 0.5) / n_rows),
                wrap_text(status, 15), fontsize=12, ha="center", va="center")
        
        # Edge density
        ax.text((3 + 0.5) / n_cols, 1 - ((row + 1 + 0.5) / n_rows),
                wrap_text(f"{edge_density:.2f}%", 10), fontsize=12, ha="center", va="center")
        
        # Conclusion
        conclusion = "Well-maintained" if status == "Clean" else "Requires cleaning"
        ax.text((4 + 0.5) / n_cols, 1 - ((row + 1 + 0.5) / n_rows),
                wrap_text(conclusion, 15), fontsize=12, ha="center", va="center")
    
    # Add scoring row at the bottom
    scoring_text = f"Total Images: {len(image_data)} | Clean Images: {clean_count} | Dirty Images: {dirty_count} | Cleanliness Percentage: {cleanliness_percentage:.2f}%"
    ax.text(0.5, 0.5 / n_rows, wrap_text(scoring_text, 60), fontsize=14, ha="center", va="center", color="black", 
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

# Function to generate cleanliness trends chart
def generate_cleanliness_chart(clean_count, dirty_count):
    categories = ["Clean", "Dirty"]
    counts = [clean_count, dirty_count]

    fig, ax = plt.subplots()
    bars = ax.bar(categories, counts, color=["green", "red"])
    ax.set_title("Cleanliness Trends", fontsize=16, weight="bold")
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.bar_label(bars, fmt='%d', fontsize=12)  # Add count labels on bars
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

    # Calculate scoring
    total_images = len(image_data)
    clean_images = sum(1 for _, status, _, _ in image_data if status == "Clean")
    dirty_images = total_images - clean_images
    cleanliness_percentage = (clean_images / total_images) * 100 if total_images > 0 else 0

    # Display results
    st.write("### Analysis Results:")
    for file_name, status, edge_density, img in image_data:
        col1, col2 = st.columns([1, 3])

        # Show the uploaded image
        with col1:
            st.image(img, caption=file_name, use_container_width=True, channels="BGR")

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

    # Generate and display cleanliness trends chart
    st.write("### Cleanliness Trends Chart:")
    trends_chart = generate_cleanliness_chart(clean_images, dirty_images)
    st.pyplot(trends_chart)
