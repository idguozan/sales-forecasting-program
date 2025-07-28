from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import upload, forecast

app = FastAPI(title="Sales Forecast API")

# CORS settings (for frontend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Add domain restriction later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(forecast.router, prefix="/forecast", tags=["Forecast"])

@app.get("/")
def root():
    return {"message": "Sales Forecast API is running"}
