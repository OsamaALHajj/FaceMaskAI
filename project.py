import os
import torch
import torchvision
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# -----------------------------
# Dataset for Mask Detection
# -----------------------------
class MaskDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms

        # Load all images and annotations
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annots = list(sorted(os.listdir(os.path.join(root, "annotations"))))

        # Ensure images and annotations match
        assert len(self.imgs) == len(self.annots), "Mismatch between images and annotations"

        # Class dictionary
        self.class_dict = {
            "with_mask": 1,
            "without_mask": 2,
            "mask_weared_incorrect": 3,
        }

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        annot_path = os.path.join(self.root, "annotations", self.annots[idx])

        # Print image being loaded (for debugging)
        print(f"✅ Loading image: {img_path}")

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Parse XML annotation
        tree = ET.parse(annot_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            label = obj.find("name").text
            if label not in self.class_dict:
                continue

            xml_box = obj.find("bndbox")
            xmin = int(xml_box.find("xmin").text)
            ymin = int(xml_box.find("ymin").text)
            xmax = int(xml_box.find("xmax").text)
            ymax = int(xml_box.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[label])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


# -----------------------------
# Transformations
# -----------------------------
def get_transform():
    return torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


# -----------------------------
# Collate function for DataLoader
# -----------------------------
def collate_fn(batch):
    return tuple(zip(*batch))


# -----------------------------
# Load Faster R-CNN model
# -----------------------------
def get_model(num_classes):
    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the classifier head with new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# -----------------------------
# Paths and dataset setup
# -----------------------------
data_root = "D:\\python\\New folder\\archive\\"
dataset = MaskDataset(data_root, transforms=get_transform())
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)


# -----------------------------
# Device setup
# -----------------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = get_model(num_classes=4)  # 3 classes + background
model.to(device)

# -----------------------------
# Optimizer setup
# -----------------------------
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# -----------------------------
# Training loop
# -----------------------------
num_epochs = 4
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for imgs, targets in data_loader:
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass and optimization
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    print(f"📢 Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# -----------------------------
# Save the trained model
# -----------------------------
torch.save(model.state_dict(), "fasterrcnn_mask_trained.pth")
print("✅ Model saved successfully.")
