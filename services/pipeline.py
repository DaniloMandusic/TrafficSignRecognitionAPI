from ultralytics import YOLO
from services.models.yolo_detect import detect_traffic_sign
from services.models.efficientnet_classification import create_classifier, transform_image, predict, get_class_label
from utils.image import decode_image, crop_roi

import cv2

def process_image(image_bytes: bytes):
    image = decode_image(image_bytes)

    detection = detect_traffic_sign(image)

    if not detection:
        return {"error": "no sign detected"}

    bbox = detection["bbox"]
    roi = crop_roi(image, bbox)

    classifier = create_classifier()
    classifier.eval()

    transformed_roi = transform_image(roi)

    predicted_class = predict(classifier, transformed_roi)

    class_label = get_class_label(predicted_class)

    return class_label