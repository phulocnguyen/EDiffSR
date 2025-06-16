import os
import random
import cv2
from tqdm import tqdm

AID_dir = '/Users/phulocnguyen/Documents/Workspace/EDiffSR/AID'

class_names = [
    'Airport', 'BareLand', 'BaseballField', 'Beach',
    'Bridge', 'Center', 'Church', 'Commercial',
    'DenseResidential', 'Desert', 'Farmland', 'Forest', 
    'Industrial', 'Meadow', 'MediumResidential', 'Mountain', 
    'Park', 'Parking', 'Playground', 'Pond',
    'Port', 'RailwayStation', 'Resort', 'River',
    'School', 'SparseResidential', 'Square', 'Stadium', 
    'StorageTanks', 'Viaduct'
]

train_save_dir = '/Users/phulocnguyen/Documents/Workspace/EDiffSR/dataset/train/GT'
test_root_dir = '/Users/phulocnguyen/Documents/Workspace/EDiffSR/dataset/test/GT'

os.makedirs(train_save_dir, exist_ok=True)

for class_name in tqdm(class_names, desc='Processing classes'):
    class_folder = os.path.join(AID_dir, class_name)
    jpg_files = [f for f in os.listdir(class_folder) if f.lower().endswith('.jpg')]



    selected_files = random.sample(jpg_files, 130)

    for i, fname in enumerate(selected_files):
        img_path = os.path.join(class_folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"annot read image: {img_path}")
            continue

        img = img[44:555, 44:555] 

        if i < 100:
            save_dir = train_save_dir
        else:
            save_dir = os.path.join(test_root_dir, class_name)
            os.makedirs(save_dir, exist_ok=True)

        new_name = fname.replace('.jpg', '.png')
        save_path = os.path.join(save_dir, new_name)
        cv2.imwrite(save_path, img)
