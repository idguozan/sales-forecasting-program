import os
import shutil
import sys
import subprocess
import json
from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

LAST_UPLOADED_FILE_JSON = os.path.join(UPLOAD_DIR, "last_uploaded.json")

ALLOWED_EXTENSIONS = {".xlsx", ".xls", ".csv", ".sql", ".sqlite", ".db"}

def save_last_uploaded_file(path: str):
    with open(LAST_UPLOADED_FILE_JSON, "w") as f:
        json.dump({"last_file": path}, f)

def get_last_uploaded_file():
    if not os.path.exists(LAST_UPLOADED_FILE_JSON):
        return None
    with open(LAST_UPLOADED_FILE_JSON) as f:
        data = json.load(f)
        return data.get("last_file")


@router.post("/upload/", summary="Upload dataset")
async def upload_dataset(file: UploadFile = File(...)):
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    save_path = os.path.join(UPLOAD_DIR, filename)

    # Write file to disk
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while saving file: {e}")

    # Save last uploaded file path
    save_last_uploaded_file(save_path)

    return {
        "message": f"{filename} uploaded successfully.",
        "path": save_path,
    }


@router.post("/start-analysis/", summary="Start analysis with last uploaded file")
async def start_analysis():
    data_path = get_last_uploaded_file()
    if not data_path:
        raise HTTPException(status_code=400, detail="Please upload a data file first.")

    run_batch_path = os.path.join(PROJECT_ROOT, "scripts", "run_batch_new.py")
    if not os.path.exists(run_batch_path):
        raise HTTPException(status_code=500, detail="run_batch_new.py not found.")

    python_exec = sys.executable

    try:
        subprocess.Popen(
            [python_exec, run_batch_path, data_path],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while starting analysis: {e}")

    return {"message": "Analysis started. Outputs are being prepared."}
