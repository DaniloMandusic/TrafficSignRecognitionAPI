from fastapi import APIRouter, UploadFile, File

router = APIRouter()

@router.get("/")
def root():
    return {"message": "Traffic sign API is running"}