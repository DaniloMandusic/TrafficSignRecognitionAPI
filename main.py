from fastapi import FastAPI
from api.routes import router

# uvicorn main:app --reload
# uvicorn main:app --reload --reload-exclude services/models/checkpoints/*
# uvicorn main:app

# taskkill /F /IM python.exe

app = FastAPI()
app.include_router(router)


