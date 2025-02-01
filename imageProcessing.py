# Image processing for multiple images in a folder
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Function to detect flat area and analyze severity
def detect_flat_area_and_severity(img, reference_mm=100.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=100, threshold2=200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x_min_combined, y_min_combined = img.shape[1], img.shape[0]
    x_max_combined, y_max_combined = 0, 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= 20:
            x, y, w, h = cv2.boundingRect(contour)
            x_min_combined = min(x_min_combined, x)
            y_min_combined = min(y_min_combined, y)
            x_max_combined = max(x_max_combined, x + w)
            y_max_combined = max(y_max_combined, y + h)

    if x_max_combined > x_min_combined and y_max_combined > y_min_combined:
        combined_width_pixels = x_max_combined - x_min_combined
        combined_height_pixels = y_max_combined - y_min_combined

        pixels_per_mm = img.shape[1] / reference_mm
        combined_width_mm = combined_width_pixels / pixels_per_mm
        combined_height_mm = combined_height_pixels / pixels_per_mm

        flat_area_pixels = np.sum(edges[y_min_combined:y_max_combined, x_min_combined:x_max_combined] == 255)
        flat_area_mm2 = (flat_area_pixels * reference_mm ** 2) / (combined_width_pixels * combined_height_pixels) if combined_width_pixels > 0 else 0

        severity = calculate_severity(flat_area_mm2)
        impact_analysis = perform_impact_analysis(severity)

        return {
            "image": img,
            "flat_area_mm2": flat_area_mm2,
            "severity": severity,
            "impact_analysis": impact_analysis,
        }
    else:
        return None

# Function to calculate severity
def calculate_severity(flat_area_mm2):
    if 1 <= flat_area_mm2 < 50:
        return "Low Severity"
    elif 50 <= flat_area_mm2 < 100:
        return "Medium Severity"
    elif flat_area_mm2 >= 100:
        return "High Severity"
    else:
        return "No Flat Area Detected"

# Function to perform impact analysis
def perform_impact_analysis(severity):
    if severity == "Low Severity":
        return "Minimal impact, no immediate action required. Normal operation."
    elif severity == "Medium Severity":
        return "Potential impact on wheel performance, recommend further inspection and monitoring."
    elif severity == "High Severity":
        return "High risk of operational failure, urgent replacement or repair required."
    else:
        return "No Flat area detected. No impact."

# Function to process and display results for all images in a folder
def process_folder(folder_path, reference_mm=100.0):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    num_images = len(image_files)

    if num_images == 0:
        print("No images found in the folder.")
        return

    results_to_display = []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path)

        if img is None:
            print(f"Error: Unable to load {image_path}")
            continue

        result = detect_flat_area_and_severity(img, reference_mm)
        if result:
            details = f"Flat Area: {result['flat_area_mm2']:.2f} mm²\n"
            details += f"Severity: {result['severity']}\n"
            details += f"Impact: {result['impact_analysis']}"
            results_to_display.append((result["image"], details))

    # Display results
    if len(results_to_display) == 0:
        print("No flat areas detected in any images.")
        return

    fig = plt.figure(figsize=(12, 6 * len(results_to_display)))  # Adjust size for all images and text
    gs = GridSpec(len(results_to_display), 2, width_ratios=[1, 1])

    for i, (img, details) in enumerate(results_to_display):
        # Left pane: Image
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.axis("off")
        ax1.set_title(f"Image {i+1}", fontsize=12)

        # Right pane: Analysis
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.text(0.1, 0.5, details, fontsize=12, wrap=True, va="center")
        ax2.axis("off")

    plt.tight_layout()
    plt.show()

# Example usage
folder_path = "D:/KP/Wabtech/FinalResult"  # Replace with your folder path
reference_wheel_width_mm = 100.0  # Known wheel width in mm
process_folder(folder_path, reference_wheel_width_mm)
