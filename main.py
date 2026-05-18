from fastapi import FastAPI
from api.routes import router

# uvicorn main:app --reload
# uvicorn main:app --reload --reload-exclude services/models/checkpoints/*
# uvicorn main:app

app = FastAPI()
app.include_router(router)


