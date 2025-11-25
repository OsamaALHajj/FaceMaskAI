import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor




# Set the dataset path and the image path.



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

# Class names
class_names = {
    1: "with_mask",
    2: "without_mask",
    3: "mask_weared_incorrect"
}

transform = T.ToTensor()


# -----------------------------
#   Open Webcam
# -----------------------------
cap = cv2.VideoCapture(0)  # 0 = default camera

if not cap.isOpened():
    print(" Cannot open camera!")
    exit()

print("Camera started... Press 'q' to quit.")


# -----------------------------
#   Real-time Loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert to PIL and Tensor
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)

    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(img_tensor)

    boxes = outputs[0]["boxes"].cpu().numpy()
    labels = outputs[0]["labels"].cpu().numpy()
    scores = outputs[0]["scores"].cpu().numpy()

    # Draw results
    for box, label, score in zip(boxes, labels, scores):
        if score < 0.5:
            continue

        x1, y1, x2, y2 = box.astype(int)
        class_name = class_names.get(label, "unknown")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame, f"{class_name} {score:.2f}", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

    cv2.imshow("Face Mask Detector - Webcam", frame)

    # Quit when pressing "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
