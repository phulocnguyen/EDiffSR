import os
import glob
import cv2

GT_folder = '/Users/phulocnguyen/Documents/Workspace/EDiffSR/dataset/train/GT'
save_LR_folder = '/Users/phulocnguyen/Documents/Workspace/EDiffSR/dataset/train/LR'
save_bicubic_folder = '/Users/phulocnguyen/Documents/Workspace/EDiffSR/dataset/train/bicubic'

os.makedirs(save_LR_folder, exist_ok=True)
os.makedirs(save_bicubic_folder, exist_ok=True)

image_paths = glob.glob(os.path.join(GT_folder, '*.png'))
up_scale = 4

if not image_paths:
    print("No PNG files found")
    exit()

print(f"Processing {len(image_paths)} images")

for i, img_path in enumerate(image_paths):
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    img_name = os.path.basename(img_path)
    height, width = img.shape[:2]
    
    im_LR = cv2.resize(img, (int(width / up_scale), int(height / up_scale)), interpolation=cv2.INTER_CUBIC)
    im_bic = cv2.resize(im_LR, (width, height), interpolation=cv2.INTER_CUBIC)
    
    cv2.imwrite(os.path.join(save_LR_folder, img_name), im_LR)
    cv2.imwrite(os.path.join(save_bicubic_folder, img_name), im_bic)
    
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(image_paths)}")

print("Done!")