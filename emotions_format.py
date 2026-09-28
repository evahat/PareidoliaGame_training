import csv
import ast
import os
from PIL import Image

#---------------------------------------------------------------------
# This script reads the metadata.csv from the FacesInThings dataset, extracts bounding boxes and emotion labels, and creates YOLO-format label files for each image. 
# The labels are saved in the "labels_emotion" directory. 
# Make sure to adjust the paths before running.
#---------------------------------------------------------------------

CSV_PATH = "FacesInThings\\metadata.csv"
IMAGE_DIR = "FacesInThings\\images"
LABEL_DIR = "labels_emotion"

os.makedirs(LABEL_DIR, exist_ok=True)

emotion_map = {
    "Neutral": 0,
    "Happy": 1,
    "Sad": 2,
    "Surprised": 3,
    "Angry": 4,
    "Scared": 5,
    "Disgusted": 6,
    "Other": 7
}

def xywh_to_yolo(box, img_w, img_h):
    x1, y1, w, h = box
    return (
        (x1 + w / 2) / img_w,
        (y1 + h / 2) / img_h,
        w / img_w,
        h / img_h
    )

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    print("CSV columns:", reader.fieldnames)

    count = 0

    for row in reader:
        filename = row.get("file")
        if not filename:
            print("Missing filename")
            continue

        image_path = os.path.join(IMAGE_DIR, filename)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        emotion = row.get("Emotion?", "").strip()
        if emotion not in emotion_map:
            print(f"Unknown emotion '{emotion}', using Other")
            emotion = "Other"

        try:
            boxes = ast.literal_eval(row.get("boxes", "[]"))
        except Exception as e:
            print(f"Bad boxes for {filename}: {e}")
            continue

        if not boxes:
            print(f"No boxes in {filename}")
            continue

        img = Image.open(image_path)
        img_w, img_h = img.size

        label_path = os.path.join(
            LABEL_DIR,
            os.path.splitext(filename)[0] + ".txt"
        )

        with open(label_path, "w") as out:
            for box in boxes:
                x, y, w, h = xywh_to_yolo(box, img_w, img_h)
                out.write(
                    f"{emotion_map[emotion]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                )

        count += 1
        print(f"Wrote {label_path}")

    print(f"\nDONE. Created {count} label files.")
