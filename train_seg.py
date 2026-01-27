from ultralytics import YOLO
import torch
import sys

def main():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if not cuda_available:
        print("WARNING: CUDA not available. Training will be slow on CPU.")
        device = 'cpu'
    else:
        print(f"Using device: {torch.cuda.get_device_name(0)}")
        device = 0

    # Load a YOLOv8s-seg model for better accuracy (upgraded from nano)
    # Alternative: 'yolo11n-seg.pt' for latest YOLO architecture
    # model_name = 'yolov8s-seg.pt'  # Change to 'yolo11n-seg.pt' or 'yolov8m-seg.pt' if available
    model_name = 'yolo11l-seg.pt'  # Change to 'yolo11n-seg.pt' or 'yolov8m-seg.pt' if available
    try:
        model = YOLO(model_name)
        print(f"Loaded model: {model_name}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Train the model with improved options
    # We use data_hold.yaml as it correctly defines the 'hold' class for the labels
    print("Starting training...")
    try:
        model.train(
            data='data_hold.yaml',
            epochs=150,      # Increased epochs for better convergence
            imgsz=800,       # Increased from 640 for better small object detection
            batch=-1,        # Auto-batch: find the largest batch size that fits in GPU memory
            device=device,   # Use CUDA if available
            optimizer='auto', # Automatically chooses the best optimizer (AdamW or SGD)
            patience=50,     # Increased patience for more training flexibility
            cos_lr=True,     # Use cosine learning rate scheduler
            lr0=0.01,        # Initial learning rate
            lrf=0.01,        # Final learning rate fraction
            dropout=0.1,     # Add dropout to prevent overfitting
            close_mosaic=10, # Disable mosaic in last 10 epochs for better accuracy
            # Augmentation improvements
            degrees=180.0,   # Rotation augmentation (holds can be any orientation)
            flipud=0.5,      # Vertical flip
            fliplr=0.5,      # Horizontal flip
            mosaic=1.0,      # Enable mosaic augmentation
            mixup=0.1,       # Mixup augmentation
            copy_paste=0.1,  # Copy-paste augmentation for segmentation masks
            scale=0.5,       # Multi-scale training (image scale +/- 50%)
            translate=0.1,   # Translation augmentation
            perspective=0.0001, # Perspective transform augmentation
            name='bouldering_holds_seg',
            project='training_runs',
            exist_ok=True
        )
        print("Training completed successfully.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return

if __name__ == '__main__':
    main()
