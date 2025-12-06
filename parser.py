import os
import glob
import json
from utils import load_config

def parse_lg_file(file_path):
    """
    Parses a single .lg file to extract labels and bounding boxes.
    
    Args:
        file_path (str): Path to the .lg file.
        
    Returns:
        list: A list of dictionaries, each containing 'label' and 'bbox'.
            bbox format is [x1, y1, x2, y2].
    """
    if not os.path.exists(file_path):
        print(f"Error: LG file not found at {file_path}")
        return []

    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    objects_map = {} # obj_id -> label
    bbox_map = {}    # obj_id -> [x1, y1, x2, y2]
    
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            continue
            
        parts = [p.strip() for p in line.split(',')]
        
        if line.startswith("O,"):
            # Format: O, Object_ID, Label, Weight, ...
            if len(parts) >= 3:
                obj_id = parts[1]
                label = parts[2]
                objects_map[obj_id] = label
                
        elif line.startswith("BB,"):
            # Format: BB, Object_ID, xmin, ymin, xmax, ymax
            if len(parts) >= 6:
                obj_id = parts[1]
                try:
                    # x1, y1, x2, y2
                    bbox = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
                    bbox_map[obj_id] = bbox
                except ValueError:
                    pass
                    
    results = []
    for obj_id, label in objects_map.items():
        if obj_id in bbox_map:
            results.append({
                'label': label,
                'bbox': bbox_map[obj_id]
            })
            
    return results

def process_dataset(lg_dir, img_dir, mapping_path="class_mapping.json", annotations_path="train_annotations.json"):
    """
    Scans all .lg files, generates class mapping, and saves all annotations to a single JSON.
    
    Args:
        lg_dir (str): Directory containing .lg files.
        img_dir (str): Directory containing image files (used for verification/path construction).
        mapping_path (str): Path to save the class_mapping.json.
        annotations_path (str): Path to save the train_annotations.json.
    """
    lg_files = glob.glob(os.path.join(lg_dir, "*.lg"))
    unique_labels = set()
    all_annotations = []
    
    print(f"Scanning {len(lg_files)} files in {lg_dir}...")
    
    for i, file_path in enumerate(lg_files):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(lg_files)} files...")
            
        # Parse the LG file
        results = parse_lg_file(file_path)
        
        # Get filename base (e.g., "01649")
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        image_filename = base_name + ".png"
        
        boxes = []
        labels = []
        
        for item in results:
            unique_labels.add(item['label'])
            boxes.append(item['bbox'])
            labels.append(item['label'])
            
        if boxes: # Only add if there are annotations
            all_annotations.append({
                "file_id": base_name,
                "image_name": image_filename,
                "boxes": boxes,
                "labels": labels
            })

    # 1. Generate and Save Class Mapping
    sorted_labels = sorted(list(unique_labels))
    class_mapping = {label: idx + 1 for idx, label in enumerate(sorted_labels)}

    
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=4)
    print(f"Saved class mapping with {len(class_mapping)} classes to {mapping_path}")

    # 2. Save Annotations for OffHME dataset
    annotations_json = {
        "source": "OffHME CROHME Training Set",
        "annotations": all_annotations
    }

    with open(annotations_path, 'w') as f:
        json.dump(annotations_json, f, indent=4)
    print(f"Saved {len(all_annotations)} annotated samples to {annotations_path}")
    
    return class_mapping, all_annotations

if __name__ == "__main__":
    config = load_config()
    lg_dir = config['paths']['train_lg_dir']
    img_dir = config['paths']['train_img_dir']
    process_dataset(lg_dir, img_dir)
