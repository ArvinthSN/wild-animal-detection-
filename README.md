# Animal Intrusion Detection - MVP Pipeline

This project provides a complete, production-ready pipeline for training and deploying an Animal Intrusion Detection model using YOLOv8, targeting >90% accuracy.

## Project Structure

- `data_manager.py`: Handles downloading the dataset from Kaggle (`antoreepjana/animals-detection-images-dataset`) and organizing it into YOLO format (converts XML to TXT if needed).
- `train_model.py`: The main training script. it:
    1.  Checks/Downloads data.
    2.  Configures the dataset.
    3.  Initializes YOLOv8n (Nano) for lightweight MVP.
    4.  Applies **Strong Augmentation** (Mosaic, Mixup, HSV) and **Focal Loss** for class imbalance.
    5.  Trains for 50 epochs (with early stopping).
    6.  Validates and saves the best model.
- `inference_pipeline.py`: A script to run the trained model on images, videos, or webcam.
- `inspect_dataset.py`: Helper to verify dataset structure.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    (Ensure `ultralytics`, `kagglehub`, `albumentations` are installed).

2.  **Dataset**:
    The dataset (9GB) will be downloaded automatically when you run the training script.

## How to Run

### 1. Train the Model
This is the main command. It handles everything.
```bash
python train_model.py
```
*Note: The first run will take time to download the 9GB dataset.*

### 2. Inference (Detection)
Once training is complete, run:
```bash
# Test on an image
python inference_pipeline.py --source path/to/image.jpg

# Test on Webcam
python inference_pipeline.py --source 0
```

## Model Strategy (Option A)

- **Model**: YOLOv8n (Nano) - Optimized for speed and accuracy (MVP).
- **Augmentation**: High degrees of geometric and photometric augmentation enabled to handle "real-world" conditions (Night, Blur, etc.).
- **Imbalance Handling**: Focal Loss (`fl_gamma=1.5`) enabled to focus on hard-to-detect animals.
- **Metric Lower Bound**: The script warns if mAP@0.5 is < 90%.

## Troubleshooting

- **Download Interrupted**: Rerunning `python train_model.py` will resume/retry the download via `kagglehub`.
- **Memory Issues**: If GPU OOM, `batch=-1` (Auto) should handle it, but you can manually set `batch=16` in `train_model.py`.
