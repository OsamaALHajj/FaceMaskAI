import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import numpy as np
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



# Set the dataset path and the image path.



def boxes_intersect(box1, box2, threshold=0.3):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return False
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou > threshold


def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=4)

model.load_state_dict(torch.load(
    r"C:\Users\DELL\Desktop\MY_file\Final project-python\FaceMaskAI\fasterrcnn_mask_trained.pth",
    map_location=device
))

model.to(device)
model.eval()

class_names = {1: "with_mask", 2: "without_mask", 3: "mask_weared_incorrect"}

img_path = r"C:\Users\DELL\Desktop\MY_file\Final project-python\FaceMaskAI\archive\images-tast\maksssksksss128.png"

img = Image.open(img_path).convert("RGB")
transform = T.ToTensor()
img_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    prediction = model(img_tensor)

boxes = prediction[0]["boxes"].cpu().numpy()
labels = prediction[0]["labels"].cpu().numpy()
scores = prediction[0]["scores"].cpu().numpy()

incorrect_mask_boxes = [
    box
    for box, label, score in zip(boxes, labels, scores)
    if label == 3 and score > 0.4
]

draw = ImageDraw.Draw(img)
font = ImageFont.load_default()

for box, label, score in zip(boxes, labels, scores):
    if score < 0.4:
        continue

    label_name = class_names.get(label, "unknown")

    if label_name == "with_mask":
        if any(boxes_intersect(box, incorrect_box) for incorrect_box in incorrect_mask_boxes):
            continue

    draw.rectangle(box, outline="red", width=2)
    draw.text((box[0], box[1] - 10), f"{label_name} {score:.2f}", fill="white", font=font)

img_cv = np.array(img)
img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

cv2.namedWindow("Detection Result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detection Result", 750, 600)
cv2.imshow("Detection Result", img_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
