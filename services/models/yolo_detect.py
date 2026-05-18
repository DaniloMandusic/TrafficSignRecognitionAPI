from ultralytics import YOLO


def detect_traffic_sign(image):
    model = YOLO("services/models/checkpoints/yolo.pt")

    results = model(image)[0]
    boxes = results.boxes
    detections = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "class_id": cls
        })

    #print(detections)

    roi = get_largest_detection(detections)
    #print(roi)

    return roi

def get_largest_detection(detections):
    if not detections:
        return None

    def area(det):
        x1, y1, x2, y2 = det["bbox"]
        return (x2 - x1) * (y2 - y1)

    return max(detections, key=area)