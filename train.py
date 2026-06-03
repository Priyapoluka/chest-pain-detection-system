import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Verify it's a PyTorch model
model._check_is_pytorch_model()

# Dataset configuration
data_yaml_path = "data.yaml"

# Train the model
model.train(
    data=data_yaml_path,
    epochs=100,
    imgsz=600,
    batch=16,
    device='cpu',
    workers=8,
    pretrained=True,
    optimizer='auto',
    verbose=True,
    plots=True
)

print("✅ Training complete! Model saved in runs/detect/train/")
