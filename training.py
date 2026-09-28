import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from torch.utils.data import ConcatDataset

#---------------------------------------------------------------------
# This script trains a Tiny YOLOv2 model to detect emotions in the FacesInThings dataset, using the background images from COCO as negative samples.
# Make sure to adjust the paths before running. The trained model will be saved as "model_final.pth".
# To follow trainig as stated in the project, first train for 50 epochs, then re-run with the line that loads "model_checkpoint.pth" for another 20 epochs.
# Make sure to change the name of the checkpoint before fine-tuning.
#---------------------------------------------------------------------


DATASET_PATH = "data\\FacesInThings\\images"
LABELS_PATH = "labels_emotion"
COCO_BG_DIR = "coco_bg\\images"

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

IMG_SIZE = 416
GRID_SIZE = 13
NUM_CLASSES = 8
NUM_ANCHORS = 5

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 20

CONF_THRESHOLD = 0.6
IOU_THRESHOLD = 0.4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Anchors (pixels relative to 416)
ANCHORS = [
    (45, 41),
    (92, 89),
    (146, 143),
    (209, 199),
    (289, 267)
]
NUM_ANCHORS = 5

def iou_wh(box1, box2):
    w1, h1 = box1
    w2, h2 = box2
    inter = min(w1, w2) * min(h1, h2)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union
def iou(box1, box2):
    # boxes: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / (area1 + area2 - inter + 1e-6)
def nms(boxes, scores, iou_thresh):
    keep = []
    idxs = np.argsort(scores)[::-1]

    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)
        idxs = idxs[1:]

        idxs = [
            i for i in idxs
            if iou(boxes[current], boxes[i]) < iou_thresh
        ]

    return keep
def decode_predictions(pred):
    boxes = []
    scores = []
    classes = []

    pred = pred.detach().cpu()

    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            for a in range(NUM_ANCHORS):
                obj = torch.sigmoid(pred[gy, gx, a, 0]).item()
                if obj < CONF_THRESHOLD:
                    continue

                tx, ty, tw, th = pred[gy, gx, a, 1:5]
                aw, ah = ANCHORS[a]

                cx = (gx + torch.sigmoid(tx).item()) / GRID_SIZE
                cy = (gy + torch.sigmoid(ty).item()) / GRID_SIZE
                w = (aw * torch.exp(tw).item()) / IMG_SIZE
                h = (ah * torch.exp(th).item()) / IMG_SIZE

                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                cls = torch.argmax(pred[gy, gx, a, 5:]).item()

                boxes.append([x1, y1, x2, y2])
                scores.append(obj)
                classes.append(cls)

    if len(boxes) == 0:
        return [], [], []

    keep = nms(boxes, scores, IOU_THRESHOLD)

    return (
        [boxes[i] for i in keep],
        [scores[i] for i in keep],
        [classes[i] for i in keep],
    )

def validate(model, loader):
    model.eval()

    detections = []   # (confidence, is_tp)
    total_gt = 0

    correct_emotion = 0
    total_emotion = 0

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)

            preds = model(imgs)

            for b in range(imgs.size(0)):
                # ----------------------
                # Decode predictions
                # ----------------------
                pred_boxes, pred_scores, pred_classes = decode_predictions(preds[b])

                # ----------------------
                # Decode GT boxes
                # ----------------------
                gt_boxes = []
                gt_emotions = []

                for gy in range(GRID_SIZE):
                    for gx in range(GRID_SIZE):
                        for a in range(NUM_ANCHORS):
                            if targets[b, gy, gx, a, 0] > 0:
                                tx, ty, tw, th = targets[b, gy, gx, a, 1:5]
                                aw, ah = ANCHORS[a]

                                cx = (gx + torch.sigmoid(tx)) / GRID_SIZE
                                cy = (gy + torch.sigmoid(ty)) / GRID_SIZE
                                w = (aw * torch.exp(tw)) / IMG_SIZE
                                h = (ah * torch.exp(th)) / IMG_SIZE

                                gt_boxes.append([
                                    cx - w / 2, cy - h / 2,
                                    cx + w / 2, cy + h / 2
                                ])

                                gt_emotions.append(
                                    torch.argmax(targets[b, gy, gx, a, 5:]).item()
                                )

                total_gt += len(gt_boxes)
                matched = [False] * len(gt_boxes)
                
                if len(gt_boxes) == 0:
                    for score in pred_scores:
                        detections.append((score, 0))
                    continue

                # ----------------------
                # Match predictions to GT
                # ----------------------
                for p_idx, (pb, score) in enumerate(zip(pred_boxes, pred_scores)):
                    best_iou = 0
                    best_gt = -1

                    for i, gt in enumerate(gt_boxes):
                        iou_val = iou(pb, gt)
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_gt = i

                    if best_iou >= IOU_THRESHOLD and best_gt >= 0 and not matched[best_gt]:
                        detections.append((score, 1))
                        matched[best_gt] = True

                        if pred_classes[p_idx] == gt_emotions[best_gt]:
                            correct_emotion += 1
                        total_emotion += 1
                        
                        
                    else:
                        detections.append((score, 0))

    # ----------------------
    # No detections guard
    # ----------------------
    if total_gt == 0 or len(detections) == 0:
        return 0.0, 0.0, 0.0, 0.0

    # ----------------------
    # Precision / Recall / AP
    # ----------------------
    detections.sort(key=lambda x: x[0], reverse=True)

    tp = np.array([d[1] for d in detections])
    fp = 1 - tp

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall_curve = tp_cum / (total_gt + 1e-6)
    precision_curve = tp_cum / (tp_cum + fp_cum + 1e-6)

    # VOC 11-point AP
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.any(recall_curve >= t):
            ap += np.max(precision_curve[recall_curve >= t])
    ap /= 11

    precision = precision_curve[-1]
    recall = recall_curve[-1]

    emotion_acc = correct_emotion / (total_emotion + 1e-6)

    return precision, recall, ap, emotion_acc
class BackgroundDataset(Dataset):
    def __init__(self, img_dir):
        self.imgs = [
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.endswith(".jpg")
        ]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx])
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img[:, :, ::-1] / 255.0

        img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1)

        # EMPTY target
        target = torch.zeros((GRID_SIZE, GRID_SIZE, NUM_ANCHORS, 5 + NUM_CLASSES))

        return img, target

class YoloDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(".jpg", ".txt"))

        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img[:, :, ::-1] / 255.0

        boxes = []

        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    cls, x, y, w, h = map(float, line.split())
                    boxes.append([cls,x, y, w, h])

        # ======================
        # AUGMENTATIONS
        # ======================
        if random.random() < 0.5:
            # Horizontal flip
            img = np.ascontiguousarray(img[:, ::-1, :])
            for i in range(len(boxes)):
                boxes[i][1] = 1.0 - boxes[i][1]

        if random.random() < 0.5:
            # Brightness / contrast
            alpha = 1.0 + random.uniform(-0.15, 0.15)
            beta = random.uniform(-0.1, 0.1)
            img = np.clip(alpha * img + beta, 0, 1)

        if random.random() < 0.5:
            # Gaussian noise
            noise = np.random.normal(0, 0.1, img.shape)
            img = np.clip(img + noise, 0, 1)

        if random.random() < 0.5:
            # Blur
            img = cv2.GaussianBlur(img, (5, 5), 0)

        img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1)

        target = torch.zeros((GRID_SIZE, GRID_SIZE, NUM_ANCHORS, 5 + NUM_CLASSES))

        for cls, x, y, w, h in boxes:
            cls = int(cls)

            gx = min(int(x * GRID_SIZE), GRID_SIZE - 1)
            gy = min(int(y * GRID_SIZE), GRID_SIZE - 1)

            ious = [
                iou_wh((w * IMG_SIZE, h * IMG_SIZE), a)
                for a in ANCHORS
            ]

            best_anchor = int(np.argmax(ious))
            for a in range(NUM_ANCHORS):
                if a != best_anchor:
                    target[gy, gx, a, 0] = 0

            tx = x * GRID_SIZE - gx
            ty = y * GRID_SIZE - gy
            tw = np.log((w * IMG_SIZE) / ANCHORS[best_anchor][0] + 1e-6)
            th = np.log((h * IMG_SIZE) / ANCHORS[best_anchor][1] + 1e-6)

            # iou scaled objectness
            gt_box = [x - w/2, y - h/2, x + w/2, y + h/2]

            aw, ah = ANCHORS[best_anchor]
            anchor_w = aw / IMG_SIZE
            anchor_h = ah / IMG_SIZE

            anchor_box = [
                x - anchor_w/2, y - anchor_h/2,
                x + anchor_w/2, y + anchor_h/2
            ]

            iou_score = iou(gt_box, anchor_box)
            target[gy, gx, best_anchor, 0] = iou_score

            target[gy, gx, best_anchor, 1:5] = torch.tensor([tx, ty, tw, th])

            # one-hot emotion class
            target[gy, gx, best_anchor, 5 + cls] = 1

        return img, target

class TinyYOLOv2(nn.Module):
    def __init__(self):
        super().__init__()

        def CBL(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.1)
            )

        self.features = nn.Sequential(
            CBL(3, 16),
            nn.MaxPool2d(2, 2),

            CBL(16, 32),
            nn.MaxPool2d(2, 2),

            CBL(32, 64),
            nn.MaxPool2d(2, 2),

            CBL(64, 128),
            nn.MaxPool2d(2, 2),

            CBL(128, 256),
            nn.MaxPool2d(2, 2),

            CBL(256, 512),
            CBL(512, 1024),
            CBL(1024, 1024),
        )

        self.pred = nn.Conv2d(
            1024,
            NUM_ANCHORS * (5 + NUM_CLASSES),
            kernel_size=1
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pred(x)

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.view(B, GRID_SIZE, GRID_SIZE, NUM_ANCHORS, 5 + NUM_CLASSES)
        return x

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.ce  = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        obj = target[..., 0] > 0
        noobj = target[..., 0] == 0

        loss_obj = self.bce(pred[..., 0][obj], target[..., 0][obj]) if obj.any() else 0.0
        loss_noobj = self.bce(pred[..., 0][noobj], target[..., 0][noobj]) if noobj.any() else 0.0

        loss_box = self.mse(pred[..., 1:5][obj], target[..., 1:5][obj]) if obj.any() else 0.0

        target_cls = torch.argmax(target[..., 5:], dim=-1)

        loss_cls = self.ce(pred[..., 5:][obj], target_cls[obj]) if obj.any() else 0.0


        return loss_obj + 5 * loss_noobj + 3 * loss_box + loss_cls

def train():
    dataset_parei = YoloDataset(DATASET_PATH,
                          LABELS_PATH)
    dataset_obj = BackgroundDataset(COCO_BG_DIR)
    
    
    num_parei = len(dataset_parei)
    indices = torch.randperm(num_parei)
    
    dataset_obj = torch.utils.data.Subset(
        dataset_obj,
        torch.randperm(len(dataset_obj))[:num_parei // 2]  # 33% background
    )

    train_indices = indices[:int(0.8 * num_parei)]
    val_indices   = indices[int(0.8 * num_parei):]

    parei_train_ds = torch.utils.data.Subset(dataset_parei, train_indices)
    parei_val_ds   = torch.utils.data.Subset(dataset_parei, val_indices)

    train_dataset = ConcatDataset([parei_train_ds, dataset_obj])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(parei_val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = TinyYOLOv2().to(DEVICE)
    
    # uncomment to continue training after initial 50 epochs
    # model.load_state_dict(torch.load("model_checkpoint.pth", map_location=DEVICE)) # <-- adjust name if you want to load a different checkpoint
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = YoloLoss()

    best_AP = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        
        
        model.train()
        total_loss = 0

        for imgs, targets in train_loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)

            preds = model(imgs)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss / len(train_loader):.4f}")
        
        
        if (epoch + 1) % 5 == 0:
            precision, recall, ap, emotion_acc = validate(model, val_loader)
            print(
                f"VAL | Precision: {precision:.3f} "
                f"Recall: {recall:.3f} "
                f"AP@0.5: {ap:.3f} "
                f"Emotion Acc: {emotion_acc:.3f}"
            )

            if best_AP < precision+emotion_acc:
                best_AP = precision+emotion_acc
                patience_counter = 0
                torch.save(model.state_dict(), "model_final.pth") # <-- adjust name if you want to save checkpoints during training
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered")
                break


if __name__ == "__main__":
    train()
