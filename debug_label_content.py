
import kagglehub
import os

path = kagglehub.dataset_download("antoreepjana/animals-detection-images-dataset")
bear_label_path = os.path.join(path, 'train', 'Bear', 'Label')

print(f"Checking {bear_label_path}")
if os.path.exists(bear_label_path):
    files = os.listdir(bear_label_path)
    print(f"Found {len(files)} files.")
    for f in files[:5]:
        print(f"File: {f}")
else:
    print("Bear/Label not found")
