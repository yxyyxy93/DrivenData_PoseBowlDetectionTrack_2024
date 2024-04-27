import torch
from pathlib import Path
from ultralytics import YOLO  # Ensure this import aligns with your YOLO import in the training script
from PIL import Image
import numpy as np


def load_model(weights_path, device='cpu'):
    # Initialize the model
    model = YOLO('custom_yolov8n.yaml').to(device)
    # Load the trained weights
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def process_image(image_path, model):
    # Load image
    image = Image.open(image_path)
    image = np.array(image)

    # Convert image to a tensor
    tensor_image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    tensor_image = tensor_image.unsqueeze(0).to(model.device)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        results = model(tensor_image)

    # Process results, for example, print or extract bounding boxes
    results = results.xyxy[0]  # Get detections for the first image in batch
    print("Detected objects:")
    for det in results:
        x1, y1, x2, y2, conf, cls = det
        print(f"Class: {cls}, Box: [{x1}, {y1}, {x2}, {y2}], Confidence: {conf}")

    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Path to the saved weights and image for testing
    weights_path = Path('yolov8n_trained.pt')
    image_path = Path('path_to_test_image.jpg')  # Update this path to your test image

    # Load the model with the trained weights
    model = load_model(weights_path, device)

    # Process a test image
    results = process_image(image_path, model)

    # You can further process 'results' if needed


if __name__ == "__main__":
    main()
