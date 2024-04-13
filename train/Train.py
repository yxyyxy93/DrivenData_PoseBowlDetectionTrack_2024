import yaml
import os
from pathlib import Path
from ultralytics import YOLO, Dataset

def main():
    # Define paths
    base_dir = Path(__file__).resolve().parent / 'data_dev'
    train_dir = base_dir / 'train'
    val_dir = base_dir / 'val'

    # YOLOv8 expects a YAML file for the dataset specification
    data_yaml = {
        'train': str(train_dir / 'images'),
        'val': str(val_dir / 'images'),
        'nc': 1,  # number of classes
        'names': ['spacecraft']  # class names
    }

    # Save the data.yaml file
    with open(base_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)

    # Load model, specifying the config and number of classes. Adjust 'yolov8n' to your specific model size (s, m, l, x, etc.)
    model = YOLO('yolov8n.yaml', nc=data_yaml['nc'], names=data_yaml['names'])

    # Training the model
    train_results = model.train(data=base_dir / 'data.yaml', imgsz=640, batch=64, epochs=50)

    # Optionally save model weights
    model.save('trained_weights.pt')

    print("Training completed and model saved as 'trained_weights.pt'.")

if __name__ == "__main__":
    main()
