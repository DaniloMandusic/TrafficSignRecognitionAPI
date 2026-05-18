import numpy as np
import cv2

def decode_image(image_bytes: bytes):
    """
    bytes -> numpy.ndarray
    image is in OpenCV - BGR format
    """

    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

def crop_roi(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return image[y1:y2, x1:x2]