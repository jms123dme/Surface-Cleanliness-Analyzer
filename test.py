import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

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

# Function to generate a summary table with images
def generate_summary_table_with_images(image_data, output_path):
    fig, ax = plt.subplots(figsize=(12, len(image_data) * 2.5))
    ax.axis("off")
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
        ax.text((3 + 0.5) / n_cols, 1 - ((row + 1 + 0.5) / n_rows), f"{edge_density:.2f}%", fontsize=10, ha="center",
                va="center")
        conclusion = "Well-maintained" if status == "Clean" else "Requires cleaning"
        ax.text((4 + 0.5) / n_cols, 1 - ((row + 1 + 0.5) / n_rows), conclusion, fontsize=10, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

@app.route("/", methods=["GET", "POST"])
def upload_images():
    if request.method == "POST":
        files = request.files.getlist("files")
        if not files:
            return "No files uploaded!", 400

        image_data = []
        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Analyze image
            status, edge_density, img = analyze_surface_cleanliness(filepath)
            if img is not None:
                image_data.append((filename, status, edge_density, img))

        # Generate summary table
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], "summary.png")
        generate_summary_table_with_images(image_data, output_path)

        return redirect(url_for("display_summary"))

    return '''
    <!doctype html>
    <title>Surface Cleanliness Analyzer</title>
    <h1>Upload Images for Analysis</h1>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="files" multiple>
        <input type="submit" value="Upload and Analyze">
    </form>
    '''

@app.route("/summary")
def display_summary():
    summary_path = os.path.join(app.config["UPLOAD_FOLDER"], "summary.png")
    if not os.path.exists(summary_path):
        return "No summary available!", 400
    return send_file(summary_path, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)
import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

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

# Function to generate a summary table with images
def generate_summary_table_with_images(image_data, output_path):
    fig, ax = plt.subplots(figsize=(12, len(image_data) * 2.5))
    ax.axis("off")
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
        ax.text((3 + 0.5) / n_cols, 1 - ((row + 1 + 0.5) / n_rows), f"{edge_density:.2f}%", fontsize=10, ha="center",
                va="center")
        conclusion = "Well-maintained" if status == "Clean" else "Requires cleaning"
        ax.text((4 + 0.5) / n_cols, 1 - ((row + 1 + 0.5) / n_rows), conclusion, fontsize=10, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

@app.route("/", methods=["GET", "POST"])
def upload_images():
    if request.method == "POST":
        files = request.files.getlist("files")
        if not files:
            return "No files uploaded!", 400

        image_data = []
        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Analyze image
            status, edge_density, img = analyze_surface_cleanliness(filepath)
            if img is not None:
                image_data.append((filename, status, edge_density, img))

        # Generate summary table
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], "summary.png")
        generate_summary_table_with_images(image_data, output_path)

        return redirect(url_for("display_summary"))

    return '''
    <!doctype html>
    <title>Surface Cleanliness Analyzer</title>
    <h1>Upload Images for Analysis</h1>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="files" multiple>
        <input type="submit" value="Upload and Analyze">
    </form>
    '''

@app.route("/summary")
def display_summary():
    summary_path = os.path.join(app.config["UPLOAD_FOLDER"], "summary.png")
    if not os.path.exists(summary_path):
        return "No summary available!", 400
    return send_file(summary_path, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)
