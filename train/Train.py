import torch
import yaml
from pathlib import Path
from ultralytics import YOLO  # Adjust this import according to your YOLO model source


def main():
    # Set device globally based on your parameter
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.device(device)
    print(f"Using device: {device}")

    # Define paths
    base_dir = Path(__file__).resolve().parent.parent / 'data_dev'
    train_dir = base_dir / 'train'
    val_dir = base_dir / 'val'

    # Prepare YAML data for YOLO configuration
    data_yaml = {
        'train': str(train_dir / 'images'),
        'val': str(val_dir / 'images'),
        'nc': 1,  # number of classes
        'names': ['spacecraft']  # class names
    }

    # Save the data.yaml file
    base_dir.mkdir(parents=True, exist_ok=True)
    with open(base_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)

    # Initialize the YOLO model with configuration
    model = YOLO('yolov8n.pt')

    # Define training parameters
    train_params = {
        "data": str(base_dir / 'data.yaml'),
        "imgsz": 1024,
        "epochs": 20,
        "batch": 64,
        "device": device,
        "project": "finetune",
        "freeze": 20,
        "plots": True
    }

    # Training the model with the specified parameters
    train_results = model.train(**train_params)

    # Save the model weights after training
    save_model(model, 'yolov8n_trained.pt')

    print("Training completed and model saved as 'yolov8n_trained.pt'.")


def save_model(model, filename):
    # Save the entire model
    torch.save(model.state_dict(), filename)


if __name__ == "__main__":
    main()
