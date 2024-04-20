import torch
import yaml
from pathlib import Path
from ultralytics import YOLO  # Make sure this import is adapted to the correct library


def main():
    # Set device globally based on your parameter
    device = 'cuda' if torch.cuda.is_available() and train_params['device'] == 'cuda' else 'cpu'
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

    # Initialize the YOLO model
    model = YOLO('yolov8n.yaml')

    # Define training parameters
    train_params = {
        "data": str(base_dir / 'data.yaml'),  # Path to your dataset configuration file
        "imgsz": 640,
        "epochs": 2,
        "fraction": 0.1,
        "batch": 64,
        "device": device,  # Use global device setting
        "project": "finetune",
        "freeze": 20,
        "plots": True
    }

    # Training the model with the specified parameters
    train_results = model.train(**train_params)

    print("Checkpoint data before saving:", model.ckpt)
    # Usage at the end of your main function
    save_model(model, 'trained_weights.pt', train_params)
    print("Training completed and model saved as 'trained_weights.pt'.")


def save_model(model, filename, train_params):
    if model.ckpt is None:
        # Manually create a checkpoint if not present
        model.ckpt = {'state_dict': model.state_dict()}
    torch.save({**model.ckpt, **train_params}, filename)


if __name__ == "__main__":
    main()
