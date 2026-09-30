from fastapi import FastAPI, UploadFile, File
import shutil
import os

app = FastAPI(title="AI Shorts Generator")

@app.get("/")
def home():
    return {"message": "AI Shorts Generator is running!"}

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {
        "status": "Video Uploaded Successfully",
        "filename": file.filename,
        "next_step": "Extracting audio and detecting highlights"
    }
