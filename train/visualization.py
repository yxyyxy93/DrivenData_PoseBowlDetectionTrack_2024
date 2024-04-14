import os
from pathlib import Path
from ultralytics import YOLO

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_image_with_bbox(image_path, results):
    img = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Iterate through all detections in the results
    for result in results:
        # Assuming 'result' is a single detection with attributes like 'xyxy', 'cls', 'conf'
        for box in result:
            left, top, right, bottom = np.array(box['xyxy']).astype(int)
            width = right - left
            height = bottom - top
            label = result.names[int(box['cls'])]  # Assuming result.names exists
            confidence = float(box['conf'])

            # Create a Rectangle patch
            rect = patches.Rectangle((left, top), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            # Optionally add text (label and confidence)
            ax.text(left, top, f'{label} {confidence:.2f}', color='white', fontsize=12,
                    bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')  # Turn off axis
    plt.show()


def load_and_visualize_model(weights_path, images_dir, results_dir):
    # Ensure the results directory exists
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Check if the custom weights file exists
    if os.path.exists(weights_path):
        # Load the model with the specified weights
        model = YOLO(weights_path)
    else:
        # Load the model with default weights, specify the correct default model configuration
        print("Weight file not found, loading default weights.")
        model = YOLO('yolov8n.yaml')  # Adjust 'yolov8n.yaml' based on your model configuration needs

    # Load images from the specified directory
    images = list(Path(images_dir).rglob('*.png'))[:5]  # Adjust the glob pattern if different image formats are used

    # Process each image
    for image_path in images:
        # Perform inference
        results = model.predict(image_path)
        # Each result in results might need individual processing
        plot_image_with_bbox(image_path, results)
        print(f"Processed and saved results for {image_path.name}")


def main():
    # Assuming the directory structure aligns with what's set in train.py
    base_dir = Path(__file__).resolve().parent.parent / 'data_dev'
    images_dir = base_dir / 'val/images'  # Assuming you want to visualize on validation images
    results_dir = base_dir / 'results'  # Directory to save visualized images

    weights_path = 'trained_weights.pt'  # Assuming weights are in the script's directory or specify the path

    load_and_visualize_model(weights_path, images_dir, results_dir)


if __name__ == "__main__":
    main()
