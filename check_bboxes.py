import os
import cv2
import glob
import random
import matplotlib.pyplot as plt

# ================= CONFIGURATION PATHS =================
BASE_PATH = "TC11_CROHME23" # CROHME 2023 OffHME dataset base path
IMG_DIR = os.path.join(BASE_PATH, "IMG", "train", "OffHME")
LG_DIR = os.path.join(BASE_PATH, "SymLG", "train", "OffHME")
# =================================================

def parse_lg_file(file_path):
    """
    Parses the .lg file to extract bounding boxes.
    Returns a list of dicts: {'label': str, 'bbox': [x1, y1, x2, y2]}
    """
    bboxes = []
    
    if not os.path.exists(file_path):
        print(f"Error: LG file not found at {file_path}")
        return []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        # We only care about lines starting with "BB"
        if line.startswith("BB"):
            parts = line.split(',')
            # Format: BB, Label_ID, xmin, ymin, xmax, ymax
            # Example: BB, T_3, 10.44, 30.07, 105.97, 135.05
            if len(parts) >= 6:
                label_id = parts[1].strip()
                try:
                    # Convert to float first, then int for pixel coordinates
                    raw_x1 = int(float(parts[2]))
                    raw_y1 = int(float(parts[3]))
                    raw_x2 = int(float(parts[4]))
                    raw_y2 = int(float(parts[5]))
                    
                    # Sort (Standardize direction, currently the LG files have inconsistent order)
                    x1 = min(raw_x1, raw_x2)
                    y1 = min(raw_y1, raw_y2)
                    x2 = max(raw_x1, raw_x2)
                    y2 = max(raw_y1, raw_y2)
                    
                    bboxes.append({
                        'label': label_id,
                        'bbox': [x1, y1, x2, y2]
                    })
                except ValueError:
                    print(f"Skipping malformed line: {line}")
                    
    return bboxes

def visualize_sample(lg_filename=None):
    """
    Visualizes one sample. If no filename provided, picks a random one.
    """
    # 1. Select a file
    if lg_filename is None:
        # Get all .lg files
        all_lg_files = glob.glob(os.path.join(LG_DIR, "*.lg"))
        if not all_lg_files:
            print(f"No .lg files found in {LG_DIR}")
            return
        selected_file = random.choice(all_lg_files)
        filename_base = os.path.basename(selected_file).replace('.lg', '')
    else:
        filename_base = lg_filename.replace('.lg', '')

    # 2. Define Paths
    lg_path = os.path.join(LG_DIR, filename_base + ".lg")
    
    # Images in OffHME
    img_path = os.path.join(IMG_DIR, filename_base + ".png")
    
    if not os.path.exists(img_path):
        print(f"Could not find image for {filename_base} in {IMG_DIR}")
        return

    # 3. Load Image
    # cv2 loads in BGR format
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to open image: {img_path}")
        return
    
    # Convert to RGB for Matplotlib display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 4. Get Bounding Boxes
    boxes = parse_lg_file(lg_path)
    print(f"Visualizing: {filename_base}")
    print(f"Found {len(boxes)} bounding boxes.")

    # 5. Draw Boxes
    for item in boxes:
        label = item['label']
        x1, y1, x2, y2 = item['bbox']
        
        # Generate random distinct color
        color = (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))
        
        # Draw Rectangle (Thickness 2)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
        
        # Draw Label Background (for readability)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_rgb, (x1, y1 - 20), (x1 + w, y1), color, -1)
        
        # Draw Text (White)
        cv2.putText(img_rgb, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 6. Show Plot
    plt.figure(figsize=(12, 12))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f"Check: {filename_base}")
    plt.show()

if __name__ == "__main__":
    for i in range(2): 
        visualize_sample()