
import kagglehub
import os

path = kagglehub.dataset_download("antoreepjana/animals-detection-images-dataset")
print(f"Dataset Path: {path}")

print("Listing first 20 jpg/xml pairs:")
count = 0
for root, dirs, files in os.walk(path):
    for f in files:
        if f.lower().endswith('.jpg'):
            # Check for xml
            base = os.path.splitext(f)[0]
            xml = base + '.xml'
            
            # Look for XML in same folder
            if xml in files:
               print(f"SAME FOLDER: {os.path.join(root, f)} -> {os.path.join(root, xml)}")
            else:
               # Look in subfolder 'Label'?
               if 'Label' in dirs:
                   label_dir = os.path.join(root, 'Label')
                   if os.path.exists(os.path.join(label_dir, xml)):
                       print(f"LABEL SUBFOLDER: {os.path.join(root, f)} -> {os.path.join(label_dir, xml)}")
                       count += 1
                       if count > 5: break
                   else:
                       # Debug where else?
                       pass
               else:
                   # Check parallel folder?
                   pass

    if count > 5: break

# Just print general structure
print("\nGeneral Structure Walk:")
for root, dirs, files in os.walk(path):
    level = root.replace(path, '').count(os.sep)
    if level > 3: continue
    print(f"{'  '*level}{os.path.basename(root)}/ (Files: {len(files)}, Dirs: {len(dirs)})")
    if level == 2: # Class level?
         # Check if XMLs are here
         xmls = [f for f in files if f.endswith('.xml')]
         print(f"    XMLs: {len(xmls)}")
         if 'Label' in dirs:
             print(f"    Has Label folder")
