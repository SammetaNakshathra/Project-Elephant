from ultralytics import YOLO

# Load YOLOv8 model (tiny model for speed; change to 'yolov8m.pt' for more accuracy)
model = YOLO("yolov8n.pt")

# Path to your image (replace with your actual file path)
image_path = "image.png"

# Run detection
results = model(image_path, imgsz=640, conf=0.25)  # imgsz=640, confidence=0.25

# Show results
results[0].show()  # opens image with bounding boxes
