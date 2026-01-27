import sys
import argparse
import torch

def check_env():
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    try:
        from ultralytics import YOLO
        print("Ultralytics YOLO version: installed")
        model = YOLO("yolov8n-seg.pt")
        print("YOLO model loading: OK")
    except Exception as e:
        print(f"YOLO check failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Bouldering Hold Detector CLI")
    parser.add_argument("command", choices=["train", "predict", "check"], 
                        help="Command to run: 'train' to start training, 'predict' to run inference, 'check' for environment check")
    
    args = parser.parse_args()

    if args.command == "check":
        check_env()
    elif args.command == "train":
        import train_seg
        train_seg.main()
    elif args.command == "predict":
        import predict_seg
        predict_seg.predict()

if __name__ == "__main__":
    main()
