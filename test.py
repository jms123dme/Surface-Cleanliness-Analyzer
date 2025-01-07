import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def analyze_surface_cleanliness(image_path, threshold=2.0):
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Image not found or unable to read the image.", 0, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
    edge_density = np.sum(edges > 0) / edges.size * 100
    status = "Clean" if edge_density < threshold else "Dirty"
    return status, edge_density, img

def list_images_in_folder(folder_path):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    return image_files

def display_table_with_images(image_data, save_path=None):
    fig, ax = plt.subplots(figsize=(12, len(image_data) * 2.5))
    ax.axis('off')
    headers = ["Image", "File Name", "Surface Condition", "Edge Density (%)", "Conclusion"]
    n_cols = len(headers)
    n_rows = len(image_data) + 1
    for col, header in enumerate(headers):
        ax.text((col + 0.5) / n_cols, 1 - 0.5 / n_rows, header, weight="bold", fontsize=12, ha="center", va="center",
                bbox=dict(boxstyle="round", facecolor="#40466e", edgecolor="white"), color="white")
    for row, (image_file, status, edge_density, img) in enumerate(image_data):
        thumbnail = cv2.resize(img, (50, 50))
        image_box = OffsetImage(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB), zoom=0.8)
        ab = AnnotationBbox(image_box, (0.1, 1 - ((row + 1 + 0.5) / n_rows)), frameon=False)
        ax.add_artist(ab)
        ax.text((1 + 0.5) / n_cols, 1 - ((row + 1 + 0.5) / n_rows), image_file, fontsize=10, ha="center", va="center")
        ax.text((2 + 0.5) / n_cols, 1 - ((row + 1 + 0.5) / n_rows), status, fontsize=10, ha="center", va="center")
        ax.text((3 + 0.5) / n_cols, 1 - ((row + 1 + 0.5) / n_rows), f"{edge_density:.2f}%", fontsize=10, ha="center", va="center")
        conclusion = "Well-maintained" if status == "Clean" else "Requires cleaning"
        ax.text((4 + 0.5) / n_cols, 1 - ((row + 1 + 0.5) / n_rows), conclusion, fontsize=10, ha="center", va="center")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Table saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing images: ").strip()
    if os.path.exists(folder_path):
        image_files = list_images_in_folder(folder_path)
        image_data = []
        for image_file in image_files:
            status, edge_density, img = analyze_surface_cleanliness(image_file)
            if img is not None:
                image_data.append((image_file, status, edge_density, img))
        save_path = os.path.join(folder_path, "summary_table.png")
        display_table_with_images(image_data, save_path=save_path)
    else:
        print("Invalid folder path. Exiting.")
