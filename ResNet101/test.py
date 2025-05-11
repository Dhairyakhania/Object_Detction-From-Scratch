import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import json
import os
from torch.optim.lr_scheduler import StepLR

# ========================
# Data transforms
# ========================
class CocoTransform:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

# ========================
# Custom Dataset
# ========================
class CocoDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        self.coco = COCO(ann_file)

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([image_id])
        target = {"boxes": boxes, "labels": labels, "image_id": image_id}

        if self._transforms:
            img, target = self._transforms(img, target)
        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

# ========================
# Model with layer4 unfrozen
# ========================
def get_model():
    backbone = resnet_fpn_backbone('resnet101', pretrained=True)

    # Freeze all layers
    for param in backbone.body.parameters():
        param.requires_grad = False

    # Unfreeze only layer4
    for param in backbone.body.layer4.parameters():
        param.requires_grad = True

    model = FasterRCNN(backbone, num_classes=91)  # COCO has 80 classes + background
    return model

# ========================
# COCO Evaluation
# ========================
def evaluate_model(model, data_loader, device):
    model.eval()
    coco_results = []

    for images, targets in tqdm(data_loader):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"].item())
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                x_min, y_min, x_max, y_max = box
                width, height = x_max - x_min, y_max - y_min
                coco_results.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x_min, y_min, width, height],
                    "score": float(score)
                })

    return coco_results

# ========================
# Main Training Loop
# ========================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = CocoDataset(
        img_folder='coco/train2017',
        ann_file='coco/annotations/instances_train2017.json',
        transforms=CocoTransform()
    )

    val_dataset = CocoDataset(
        img_folder='coco/val2017',
        ann_file='coco/annotations/instances_val2017.json',
        transforms=CocoTransform()
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = get_model().to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_train_loss += losses.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_val_loss += losses.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        scheduler.step()

    # Evaluation
    results = evaluate_model(model, val_loader, device)

    with open("coco_predictions.json", "w") as f:
        json.dump(results, f)

    coco_gt = COCO('coco/annotations/instances_val2017.json')
    coco_dt = coco_gt.loadRes("coco_predictions.json")

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mean_ap = coco_eval.stats[0]
    ap50 = coco_eval.stats[1]
    ap75 = coco_eval.stats[2]
    recall = coco_eval.stats[6]

    print(f"\nmAP@[0.5:0.95]: {mean_ap:.4f}")
    print(f"AP@0.50: {ap50:.4f}")
    print(f"AP@0.75: {ap75:.4f}")
    print(f"Recall@[0.5:0.95]: {recall:.4f}")

if __name__ == "__main__":
    train()
