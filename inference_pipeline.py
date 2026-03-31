
from ultralytics import YOLO
import cv2
import argparse
import os

def run_inference(source, model_path='runs/detect/animal_intrusion_v1/weights/best.pt', conf_thresh=0.5):
    if not os.path.exists(model_path):
        print(f"Model weights not found at {model_path}. Using 'yolov8n.pt' for demo.")
        model_path = 'yolov8n.pt'
        
    model = YOLO(model_path)
    
    # Run inference
    # source can be path to image, video, or '0' for webcam
    results = model.predict(source, conf=conf_thresh, save=True, show=True)
    
    for result in results:
        boxes = result.boxes
        if len(boxes) > 0:
            print(f"Intrusion Detected! Found {len(boxes)} animals.")
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                print(f" - {cls_name} ({conf:.2f})")
        else:
            print("No intrusion detected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='test_image.jpg', help='Image path, video path, or webcam id')
    parser.add_argument('--model', type=str, default='runs/detect/animal_intrusion_v1/weights/best.pt', help='Path to .pt model')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    args = parser.parse_args()
    
    run_inference(args.source, args.model, args.conf)
