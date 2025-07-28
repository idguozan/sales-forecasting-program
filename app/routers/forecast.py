import subprocess
from fastapi import APIRouter
from fastapi.responses import FileResponse
import os

router = APIRouter()

REPORTS_DIR = os.path.join("app", "static", "reports")

@router.post("/run")
def run_forecast():
    try:
        # Run run_batch.py
        subprocess.run(["python3", "scripts/run_batch.py"], check=True)
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
