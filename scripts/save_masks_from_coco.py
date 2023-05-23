import os
import json
from PIL import Image
import numpy as np

# Function to save the masks
def save_masks_to_tiff(json_path, masks_dir):
    with open(json_path, 'r') as f:
        annotations = json.load(f)['annotations']
    for annotation in annotations:
        mask_stack = np.stack(annotation['masks'], axis=-1)  # stack along the channel axis
        mask_path = os.path.join(masks_dir, f"{annotation['image_name']}_mask.tif")
        Image.fromarray(mask_stack).save(mask_path)

# Modify the COCO JSON
def modify_coco_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    for im in data['images']:
        im['mask_path'] = os.path.join("tiled_masks", f"{im['file_name']}_mask.tif")

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

# Assuming your masks directory and JSON file paths are as follows
masks_dir = "/home/work/slickformer/data/partitions/train_tiles_context_0/tiled_masks"
json_path = "//home/work/slickformer/data/partitions/train_tiles_context_0/instances_CeruleanCOCO_copy.json"

# Here's how you can use these functions and the DataPipe
save_masks_to_tiff(json_path, masks_dir)
modify_coco_json(json_path)

