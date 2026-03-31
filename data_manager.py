
import kagglehub
import os
import shutil
import yaml
from pathlib import Path
import random
from tqdm import tqdm
import glob

DATASET_NAME = "antoreepjana/animals-detection-images-dataset"
LOCAL_DATA_DIR = os.path.join(os.getcwd(), "dataset")
YOLO_DATA_YAML = os.path.join(LOCAL_DATA_DIR, "data.yaml")

def download_data():
    print("Checking/Downloading dataset via KaggleHub...")
    path = kagglehub.dataset_download(DATASET_NAME)
    print(f"Dataset available at: {path}")
    return path

def setup_yolo_structure(source_path):
    """
    Organizes data into YOLO structure:
    dataset/
      images/
        train/
        val/
      labels/
        train/
        val/
      data.yaml
    """
    if os.path.exists(LOCAL_DATA_DIR) and os.path.exists(YOLO_DATA_YAML):
        print("Dataset directory appears to be set up. Skipping organization.")
        return YOLO_DATA_YAML
    
    print("Organizing dataset into YOLO structure...")
    os.makedirs(os.path.join(LOCAL_DATA_DIR, "images/train"), exist_ok=True)
    os.makedirs(os.path.join(LOCAL_DATA_DIR, "images/val"), exist_ok=True)
    os.makedirs(os.path.join(LOCAL_DATA_DIR, "labels/train"), exist_ok=True)
    os.makedirs(os.path.join(LOCAL_DATA_DIR, "labels/val"), exist_ok=True)

    # INSPECT SOURCE first to decide how to move files
    # This logic assumes a typical structure. 
    # I will rely on the verify step to properly implement this after seeing the structure.
    # For now, I'll implement a 'smart' finder that looks for image/label pairs.
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    all_images = []
    
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                all_images.append(os.path.join(root, file))
    
    print(f"Found {len(all_images)} images.")
    if len(all_images) == 0:
        raise ValueError("No images found in dataset path!")

    # Shuffle and Split
    random.seed(42)
    random.shuffle(all_images)
    
    split_idx = int(len(all_images) * 0.85) # 85% Train, 15% Val
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    # Helper to convert XML to YOLO (normalized xywh)
    def convert_xml_to_yolo(xml_file, width, height, class_mapping):
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_file)
        root = tree.getroot()
        yolo_lines = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in class_mapping:
                continue
            cls_id = class_mapping[name]
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            
            yolo_lines.append(f"{cls_id} {x_center} {y_center} {w} {h}")
        return yolo_lines

    # Detect Classes from Folders
    print("Detecting classes from directory structure...")
    classes = sorted([d for d in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, d))])
    
    # Fix for datasets with 'train'/'test' top-level folders
    if 'train' in classes and os.path.isdir(os.path.join(source_path, 'train')):
        print("Detected 'train' folder. looking inside for classes...")
        train_path = os.path.join(source_path, 'train')
        classes = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])

    # If no folders found (flat structure), try to infer from data or use default?
    # Actually, for object detection, classes are usually in the annotation file.
    # But if structure is 'Dataset/Lion/...' assume 'Lion' is a class.
    if not classes:
        # Fallback: Check if there's a 'classes.txt'
        if os.path.exists(os.path.join(source_path, 'classes.txt')):
             with open(os.path.join(source_path, 'classes.txt'), 'r') as f:
                 classes = [line.strip() for line in f.readlines() if line.strip()]
    
    if not classes:
        # Last resort: Scan a few XMLs? or Assume predefined for this dataset?
        # User search said 80/90 classes.
        # Let's hope folders exist.
        print("Warning: No class folders found. Assuming flat structure. Class map building might fail if not provided.")
        classes = ['animal'] # Fallback
    
    class_mapping = {c: i for i, c in enumerate(classes)}
    print(f"Detected {len(classes)} classes: {classes[:5]}...")

    def process_files(image_list, split_name):
        print(f"Processing {split_name} data...")
        from PIL import Image
        
        for img_path in tqdm(image_list):
            img_name = os.path.basename(img_path)
            shutil.copy2(img_path, os.path.join(LOCAL_DATA_DIR, "images", split_name, img_name))
            
            # Identify Class from parent folder if relevant
            # parent_folder = os.path.basename(os.path.dirname(img_path))
            
            # Find Label
            label_name_txt = os.path.splitext(img_name)[0] + ".txt"
            label_name_xml = os.path.splitext(img_name)[0] + ".xml"
            
            # Search logic...
            parent = os.path.dirname(img_path)
            
            # Check for TXT
            src_txt = os.path.join(parent, label_name_txt)
            # Check for parallel 'labels' folder or similar
            if not os.path.exists(src_txt):
                # Try heuristics
                pass # (Simpler logic here for brevity, assuming adjacent or nearby)
            
            # Check for XML
            src_xml = os.path.join(parent, label_name_xml)
            if not os.path.exists(src_xml):
                 src_xml = os.path.join(parent, 'Label', label_name_xml)
            
            # Check for Custom TXT in Label folder (ClassName Xmin Ymin Xmax Ymax)
            src_txt_custom = os.path.join(parent, 'Label', label_name_txt)

            final_txt_path = os.path.join(LOCAL_DATA_DIR, "labels", split_name, label_name_txt)

            if os.path.exists(src_txt):
                shutil.copy2(src_txt, final_txt_path)
            elif os.path.exists(src_txt_custom):
                # Convert Custom TXT
                try:
                    with Image.open(img_path) as img:
                        w_img, h_img = img.size
                    
                    yolo_lines = []
                    with open(src_txt_custom, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            full_line = line.strip()
                            if not full_line: continue
                            
                            # Match class name (Longest first)
                            matched_class = None
                            for c in sorted(classes, key=len, reverse=True):
                                if full_line.startswith(c):
                                    matched_class = c
                                    break
                            
                            if matched_class:
                                cls_id = class_mapping[matched_class]
                                # Parse coordinates
                                remaining = full_line[len(matched_class):].strip()
                                coords = list(map(float, remaining.split()))
                                
                                if len(coords) == 4:
                                    xmin, ymin, xmax, ymax = coords
                                    
                                    # Normalize to YOLO (x_center, y_center, w, h)
                                    x_center = ((xmin + xmax) / 2) / w_img
                                    y_center = ((ymin + ymax) / 2) / h_img
                                    w = (xmax - xmin) / w_img
                                    h = (ymax - ymin) / h_img
                                    
                                    # Clamp to 0-1 just in case
                                    x_center = max(0, min(1, x_center))
                                    y_center = max(0, min(1, y_center))
                                    w = max(0, min(1, w))
                                    h = max(0, min(1, h))
                                    
                                    yolo_lines.append(f"{cls_id} {x_center} {y_center} {w} {h}")
                    
                    if yolo_lines:
                        with open(final_txt_path, 'w') as f:
                            f.write("\n".join(yolo_lines))
                            
                except Exception as e:
                    print(f"Error converting {src_txt_custom}: {e}")

            elif os.path.exists(src_xml):
                # CONVERT XML
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                    yolo_data = convert_xml_to_yolo(src_xml, w, h, class_mapping)
                    if yolo_data:
                        with open(final_txt_path, 'w') as f:
                            f.write("\n".join(yolo_data))
                except Exception as e:
                    print(f"Error converting {src_xml}: {e}")
            else:
                 pass

    process_files(train_images, "train")
    process_files(val_images, "val")
    
    yaml_content = {
        'path': LOCAL_DATA_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes),
        'names': classes
    }
    
    with open(YOLO_DATA_YAML, 'w') as f:
        yaml.dump(yaml_content, f)
        
    return YOLO_DATA_YAML

if __name__ == "__main__":
    src = download_data()
    # setup_yolo_structure(src) # Don't run automatically yet until inspection
