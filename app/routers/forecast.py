import subprocess
from fastapi import APIRouter
from fastapi.responses import FileResponse
import os

router = APIRouter()

# Use absolute path for reports directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "app", "static", "reports")

# Ensure reports directory exists
os.makedirs(REPORTS_DIR, exist_ok=True)

@router.post("/run")
def run_forecast():
    try:
        # Run run_batch_new.py (correct script path)
        subprocess.run(["python3", "scripts/run_batch_new.py"], check=True)
        return {
            "message": "Forecast completed",
            "pdf_report": f"/forecast/download/pdf",
            "csv_forecast": f"/forecast/download/csv"
        }
    except subprocess.CalledProcessError as e:
        return {"error": str(e)}

@router.get("/download/pdf")
def download_pdf():
    pdf_path = os.path.join(REPORTS_DIR, "forecast_report.pdf")
    if not os.path.exists(pdf_path):
        return {"error": "PDF not found"}
    return FileResponse(pdf_path, media_type="application/pdf", filename="forecast_report.pdf")

@router.get("/download/csv")
def download_csv():
    csv_path = os.path.join(REPORTS_DIR, "future_forecast.csv")
    if not os.path.exists(csv_path):
        return {"error": "CSV not found"}
    return FileResponse(csv_path, media_type="text/csv", filename="future_forecast.csv")
