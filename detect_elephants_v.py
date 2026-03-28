from ultralytics import YOLO
import cv2

# Load YOLOv8 model (pretrained on COCO)
model = YOLO("yolov8n.pt")  # tiny model for speed; use "yolov8m.pt" for better accuracy

# Path to your video
video_path = "elephants.mp4"  # replace with your video file

# Open the video
cap = cv2.VideoCapture(video_path)

# Get video properties for saving output
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_detected.mp4', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection on the frame
    results = model(frame, imgsz=640, conf=0.25)

    # Render results on the frame
    annotated_frame = results[0].plot()

    # Write frame to output video
    out.write(annotated_frame)

    # Optional: show live detection
    cv2.imshow('Elephant Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
