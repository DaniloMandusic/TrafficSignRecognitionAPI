from fastapi import APIRouter, UploadFile, File
from services.pipeline import process_image

router = APIRouter()

@router.get("/")
def root():
    return {"message": "Traffic sign API is running"}

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    result = process_image(image_bytes)

    return {
        "name": file.filename,
        "type": file.content_type,
        "result": result,
    }