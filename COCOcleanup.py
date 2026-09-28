import json
import os
import shutil

#--------------------------------------------------------------------
# This script removes all images containing persons from the COCO validation set and copies the remaining background images to a new folder.
# Make sure to adjust the paths before running. The output will be in "coco_bg/images".
#--------------------------------------------------------------------

COCO_IMG_DIR = "coco\\val2017"
ANNOTATIONS = "coco\\annotations_trainval2017\\annotations\\instances_val2017.json"
OUT_DIR = "coco_bg/images"

os.makedirs(OUT_DIR, exist_ok=True)

with open(ANNOTATIONS) as f:
    coco = json.load(f)

# Find person category id
person_id = None
for cat in coco["categories"]:
    if cat["name"] == "person":
        person_id = cat["id"]

# Images that contain persons
images_with_person = set(
    ann["image_id"]
    for ann in coco["annotations"]
    if ann["category_id"] == person_id
)

# Copy images WITHOUT persons
count = 0
for img in coco["images"]:
    if img["id"] not in images_with_person:
        src = os.path.join(COCO_IMG_DIR, img["file_name"])
        dst = os.path.join(OUT_DIR, img["file_name"])
        shutil.copy(src, dst)
        count += 1

print(f"Copied {count} background images")
