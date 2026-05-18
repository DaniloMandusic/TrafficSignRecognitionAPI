from ultralytics import YOLO
from services.models.yolo_detect import detect_traffic_sign
from utils.image import decode_image, crop_roi

import cv2

def process_image(image_bytes: bytes):
    image = decode_image(image_bytes)

    detection = detect_traffic_sign(image)

    if not detection:
        return {"error": "no sign detected"}

    bbox = detection["bbox"]
    roi = crop_roi(image, bbox)

    return "test1"