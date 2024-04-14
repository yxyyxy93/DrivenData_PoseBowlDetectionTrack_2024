import os
from pathlib import Path
from ultralytics import YOLO


def load_and_visualize_model(weights_path, images_dir, results_dir):
    # Ensure the results directory exists
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Load the model with the specified weights
    model = YOLO(weights_path)

    # Load images from the specified directory
    images = list(Path(images_dir).rglob('*.png'))  # Adjust the glob pattern if different image formats are used

    # Process each image
    for image_path in images:
        # Perform inference
        results = model.predict(image_path)

        # Save or show results
        results.show()  # This will display the image window; may not work well in non-GUI environments
        results.save(save_dir=results_dir)  # Save the visualized output to the results directory

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
