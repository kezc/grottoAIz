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

    # Load YOLO11m-seg model for balance of high accuracy and reasonable training time
    model_name = 'yolo11m-seg.pt'
    try:
        model = YOLO(model_name)
        print(f"Loaded model: {model_name}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Train on the new 1,873-image segmentation dataset
    print("Starting training on climbing_hold_seg_yolo dataset...")
    try:
        model.train(
            data='datasets/climbing_hold_seg_yolo/data.yaml',
            epochs=100,
            imgsz=1024,       # High resolution for tiny foot chips & crimps
            batch=4,          # Set to 4 to stay strictly within 12GB dedicated VRAM (prevents slow RAM paging)
            device=device,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            patience=20,
            cos_lr=True,
            close_mosaic=10,
            # Climbing-specific augmentations
            degrees=25.0,     # Realistic tilt (avoids inverted walls)
            flipud=0.0,       # Climbing walls remain upright
            fliplr=0.5,       # Horizontal flip
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.15,
            hsv_s=0.5,        # Robust to chalk dust & saturation changes
            hsv_v=0.4,        # Robust to gym lighting variations
            scale=0.5,
            translate=0.1,
            perspective=0.0005,
            name='climbing_holds_v2',
            project='training_runs',
            exist_ok=True
        )
        print("Training completed successfully.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return

if __name__ == '__main__':
    main()
