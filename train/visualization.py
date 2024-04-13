import os
from pathlib import Path
from ultralytics import YOLO

def load_and_visualize_model(weights_path, images_dir, results_dir):
    # Ensure the results directory exists
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Load the model with the specified weights
    model = YOLO(weights_path)

    # Load images from the specified directory
    images = list(Path(images_dir).rglob('*.jpg'))  # Adjust the glob pattern if different image formats are used

    # Process each image
    for image_path in images:
        # Perform inference
        results = model.predict(image_path)

        # Save or show results
        results.show()  # This will display the image window; may not work well in non-GUI environments
        results.save(save_dir=results_dir)  # Save the visualized output to the results directory

        print(f"Processed and saved results for {image_path.name}")

def main():
    weights_path = 'trained_weights.pt'  # Path to the model weights file
    images_dir = 'path/to/images'  # Directory containing images to process
    results_dir = 'path/to/results'  # Directory where the visualized images will be saved

    load_and_visualize_model(weights_path, images_dir, results_dir)

if __name__ == "__main__":
    main()
