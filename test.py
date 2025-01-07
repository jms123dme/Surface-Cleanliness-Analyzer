import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_surface_cleanliness(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # Apply edge detection using Canny
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
    
    # Calculate the percentage of detected edges
    edge_density = np.sum(edges > 0) / edges.size * 100
    
    # Display the image for visual debugging (Optional)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 2)
    plt.title("Grayscale")
    plt.imshow(gray, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title("Edges Detected")
    plt.imshow(edges, cmap='gray')
    plt.show()
    
    # Threshold for cleanliness
    if edge_density < 2.0:  # Edge density under 2% implies a clean surface
        return "The surface appears clean."
    else:
        return "The surface appears dirty or has smudges."

# Example usage
image_path = "C:\Users\Rohit Chandaliya\Downloads\Documents\hellotest.jpg"  # Replace with your image path
result = analyze_surface_cleanliness(image_path)
print(result)
