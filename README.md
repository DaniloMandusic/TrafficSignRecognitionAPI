# Traffic Sign Recognition API

## About

Traffic Sign Recognition API is a Python backend application built with FastAPI for detecting and classifying traffic signs from uploaded images.

The application uses a two-stage machine learning pipeline:

1. Traffic sign detection using YOLO
2. Traffic sign classification using EfficientNet

Both models were retrained on traffic sign datasets in separate projects:

- [YOLO Traffic Sign Detection](https://github.com/DaniloMandusic/Traffic-Sign-Detection)
- [Efficientnet Traffic Sign Recognition](https://github.com/DaniloMandusic/Traffic-Sign-Recognition)

---

## Features

- Traffic sign detection from uploaded images
- 43-class traffic sign classification
- FastAPI REST API
- Swagger/OpenAPI documentation

---

## API Endpoints

### POST `/predict`

Detects and classifies the most prominent traffic sign in the uploaded image.

### Request

`multipart/form-data`

| Field | Type | Content |
|------|------|------|
| file | File | .jpg/.png image |


## How to run
- uvicorn main:app
- YOLO and Efficientnet model checkpoints are not in this repository, you can find them in linked projects

## Project Structure

```text
project/
├── api/
│   ├── routes.py                  # router for API calls
│
├── services/
│   ├── models/
│   │   ├── checkpoints/           # model checkpoint files
│   │   │
│   │   ├── efficientnet_classification.py  # classification logic + utils
│   │   └── yolo_detect.py                  # object detection logic + utils
│   │
│   └── pipeline.py                # main pipeline (detection + classification)
│
├── utils/
│   └── image.py                   # image utilities (cv2 / preprocessing)
│
├── main.py                        # FastAPI app setup
└── README.md
```




