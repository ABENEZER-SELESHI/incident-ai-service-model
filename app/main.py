# INCIDENT-AI-SERVICE/app/main.py
from fastapi import FastAPI, UploadFile, File
import os

from app.inference import predict
from app.utils import save_upload, allowed_extension

app = FastAPI()

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/predict")
async def predict_incident(file: UploadFile = File(...)):

    if not allowed_extension(file.filename):
        return {"error": "Invalid file type"}

    file_path = save_upload(file)

    result = predict(file_path)

    return result