                            
import kagglehub
import os

def inspect_dataset():
    print("Downloading dataset...")
    path = kagglehub.dataset_download("antoreepjana/animals-detection-images-dataset")
    print(f"Dataset downloaded to: {path}")

    print("\nDirectory Structure:")
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        # Print first 5 files
        for f in files[:5]:
            print(f"{subindent}{f}")
        if len(files) > 5:
            print(f"{subindent}... ({len(files)-5} more files)")

if __name__ == "__main__":
    inspect_dataset()
