import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import tkinter as tk
from tkinter import simpledialog

def analyze_surface_cleanliness(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Image not found or unable to read the image.", 0, None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # Apply edge detection using Canny
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
    
    # Calculate the percentage of detected edges
    edge_density = np.sum(edges > 0) / edges.size * 100

    # Threshold for cleanliness
    if edge_density < 2.0:  # Edge density under 2% implies a clean surface
        status = "Clean"
    else:
        status = "Dirty"
    
    return status, edge_density, img

def generate_summary_table_with_images(folder_path):
    # List all image files in the folder
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')  # Supported image formats
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print("No images found in the folder!")
        return

    # Collect data for the table
    image_data = []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        status, edge_density, img = analyze_surface_cleanliness(image_path)
        if img is None:
            print(f"Error processing image: {image_file}")
            continue
        # Consider dirty if the surface is not aligned (edge_density > 2% implies this)
        if edge_density > 2.0:
            status = "Dirty"
        image_data.append((image_file, status, edge_density, img))

    # Display the table with images
    display_table_with_images(image_data)

def display_table_with_images(image_data):
    # Create a figure for the table
    fig, ax = plt.subplots(figsize=(12, len(image_data) * 2.5))
    ax.axis('off')

    # Define the table headers
    headers = ["Image", "File Name", "Surface Condition", "Edge Density (%)", "Conclusion"]

    # Draw a table border
    ax.plot([0, 1], [1, 1], color="black", lw=1)  # Top border
    ax.plot([0, 1], [0, 0], color="black", lw=1)  # Bottom border
    ax.plot([0, 0], [0, 1], color="black", lw=1)  # Left border
    ax.plot([1, 1], [0, 1], color="black", lw=1)  # Right border

    # Create a blank table layout
    col_widths = [0.2, 0.2, 0.2, 0.2, 0.2]
    n_cols = len(headers)
    n_rows = len(image_data) + 1

    # Add headers to the table
    for col, header in enumerate(headers):
        ax.text(
            x=(col + 0.5) / n_cols,
            y=1 - (0.5 / n_rows),
            s=header,
            weight="bold",
            fontsize=12,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", facecolor="#40466e", edgecolor="white"),
            color="white",
        )
        ax.plot([col / n_cols, (col + 1) / n_cols], [1, 1], color="black", lw=1)  # Horizontal line

    # Add data rows
    for row, (image_file, status, edge_density, img) in enumerate(image_data):
        # Column 1: Image thumbnail
        thumbnail = cv2.resize(img, (50, 50))
        image_box = OffsetImage(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB), zoom=0.8)
        ab = AnnotationBbox(
            image_box,
            ((0.5 / n_cols), 1 - ((row + 1 + 0.5) / n_rows)),
            frameon=False,
            box_alignment=(0.5, 0.5)
        )
        ax.add_artist(ab)

        # Column 2: File Name
        ax.text(
            x=(1 + 0.5) / n_cols,
            y=1 - ((row + 1 + 0.5) / n_rows),
            s=image_file,
            fontsize=10,
            ha="center",
            va="center",
        )

        # Column 3: Surface Condition
        ax.text(
            x=(2 + 0.5) / n_cols,
            y=1 - ((row + 1 + 0.5) / n_rows),
            s=status,
            fontsize=10,
            ha="center",
            va="center",
        )

        # Column 4: Edge Density (%)
        ax.text(
            x=(3 + 0.5) / n_cols,
            y=1 - ((row + 1 + 0.5) / n_rows),
            s=f"{edge_density:.2f}%",
            fontsize=10,
            ha="center",
            va="center",
        )

        # Column 5: Conclusion
        conclusion = "Well-maintained" if status == "Clean" else "Requires cleaning"
        ax.text(
            x=(4 + 0.5) / n_cols,
            y=1 - ((row + 1 + 0.5) / n_rows),
            s=conclusion,
            fontsize=10,
            ha="center",
            va="center",
        )

        # Draw horizontal lines between rows
        ax.plot([0, 1], [1 - ((row + 1) / n_rows), 1 - ((row + 1) / n_rows)], color="black", lw=1)

    # Draw vertical lines
    for col in range(1, n_cols):
        ax.plot(
            [col / n_cols, col / n_cols], [0, 1], color="black", lw=1
        )  # Vertical lines

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create a popup to get the folder path
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_path = simpledialog.askstring("Folder Path", "Enter the path to the folder containing images:")

    if folder_path:
        generate_summary_table_with_images(folder_path)
    else:
        print("No folder path provided. Exiting.")
