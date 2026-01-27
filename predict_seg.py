import csv
import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO


def predict():
    # Path to the best model weights after training
    model_path = 'runs/segment/training_runs/bouldering_holds_seg/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Trying alternative paths...")
        alternatives = [
            'training_runs/bouldering_holds_seg/weights/best.pt',
            'weights/best.pt',
            'best.pt'
        ]
        for alt in alternatives:
            if os.path.exists(alt):
                model_path = alt
                print(f"Found model at {model_path}")
                break
        else:
            print("Could not find best.pt. Please ensure training completed successfully.")
            return

    # Load the model
    model = YOLO(model_path)

    # Path to test images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    test_images = []
    for ext in extensions:
        test_images.extend(glob.glob(os.path.join('test/images', ext)))
        test_images.extend(glob.glob(os.path.join('test/images', ext.upper())))
    
    if not test_images:
        print("No test images found in test/images/")
        return

    # Create output directory for custom contour images
    save_dir = os.path.join('predictions', 'bouldering_holds_contours')
    os.makedirs(save_dir, exist_ok=True)

    # Create CSV file to save metrics
    csv_path = os.path.join(save_dir, 'detection_metrics.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Image', 'Hold_ID', 'Confidence', 'Area_pixels', 'Centroid_X', 'Centroid_Y'])

    # Run prediction on test images
    print(f"Running prediction on {len(test_images)} images...")
    
    import torch
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    results = model.predict(
        source=test_images,
        save=False,          # We will handle saving with only contours
        device=device,
        conf=0.20,           # Lowered confidence threshold to detect more holds
        iou=0.7,             # IOU threshold for NMS to reduce overlapping detections
        augment=True,        # Test-time augmentation for better accuracy
        imgsz=800            # Match training image size
    )

    total_holds = 0
    min_hold_area = 100  # Minimum area in pixels to filter out noise

    for r in results:
        # Start with a copy of the original image
        img = r.orig_img.copy()

        hold_count = 0
        img_name = os.path.basename(r.path)

        # Check if any masks were detected
        if r.masks is not None and r.boxes is not None:
            # Get confidence scores
            confidences = r.boxes.conf.cpu().numpy()

            # Iterate through each detected mask's coordinates (segments)
            for idx, (seg, conf) in enumerate(zip(r.masks.xy, confidences)):
                if len(seg) > 0:
                    # Convert coordinates to integer format for OpenCV
                    pts = np.array(seg, np.int32).reshape((-1, 1, 2))

                    # Calculate area and centroid
                    area = cv2.contourArea(pts)
                    M = cv2.moments(pts)

                    # Filter out very small detections (likely noise)
                    if area < min_hold_area:
                        continue

                    # Calculate centroid
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                    else:
                        cx, cy = 0, 0

                    hold_count += 1
                    total_holds += 1

                    # Draw the contour (color intensity based on confidence)
                    color_intensity = int(255 * conf)
                    cv2.polylines(img, [pts], isClosed=True, color=(0, color_intensity, 0), thickness=2)

                    # Add confidence score near the hold
                    cv2.putText(img, f"{conf:.2f}", (cx-20, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    # Write metrics to CSV
                    csv_writer.writerow([img_name, hold_count, f"{conf:.4f}",
                                        int(area), cx, cy])

        # Add hold count to the image
        cv2.putText(img, f"Holds: {hold_count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Get save path (img_name already set above)
        save_path = os.path.join(save_dir, img_name)
        
        # Save the result image with contours
        cv2.imwrite(save_path, img)
        print(f"Processed: {img_name} - Found {hold_count} holds")

        # Display the result
        window_name = f"Result: {img_name} (Press 'q' to quit, any key for next)"
        
        # Resize for display if the image is too large
        display_img = img.copy()
        max_width = 1280
        max_height = 720
        h, w = display_img.shape[:2]
        
        scale = min(max_width / w, max_height / h)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            display_img = cv2.resize(display_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, display_img)
        
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(window_name)
        
        if key == ord('q'):
            print("Stopping prediction early.")
            break

    # Close CSV file
    csv_file.close()

    print(f"\nSummary:")
    print(f"Total images processed: {len(results)}")
    print(f"Total holds detected: {total_holds}")
    print(f"Average holds per image: {total_holds/len(results):.1f}" if len(results) > 0 else "")
    print(f"Results saved to: {save_dir}")
    print(f"Metrics saved to: {csv_path}")

if __name__ == '__main__':
    predict()
