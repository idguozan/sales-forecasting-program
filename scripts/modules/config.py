#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration module for the sales forecasting system.
Contains all constants, parameters, and configuration settings.
"""

from pathlib import Path

# ============================================================================
# Project Configuration
# ============================================================================

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Application static directories (served by FastAPI)
APP_STATIC_DIR = PROJECT_ROOT / "app" / "static"
PLOTS_DIR = APP_STATIC_DIR / "plots"
REPORTS_DIR = APP_STATIC_DIR / "reports"

# Uploads folder (populated by FastAPI upload router)
UPLOADS_DIR = PROJECT_ROOT / "uploads"

# Create directories if they don't exist
for dir_path in [PLOTS_DIR, REPORTS_DIR, UPLOADS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Forecasting Parameters
# ============================================================================

FORECAST_HORIZON_WEEKS = 12
TARGET_COL = "total_sales"  # Changed from total_sales to total_quantity (kg)

# Enhanced feature columns
FEATURE_COLS = [
    "week_of_year", "month", "quarter", "day_of_year",
    "rolling_mean_4", "rolling_mean_8", "rolling_mean_12",
    "rolling_std_4", "rolling_std_8", "rolling_median_4",
    "lag_1", "lag_2", "lag_4", "lag_8",
    "diff_1", "diff_2", "pct_change_1", "pct_change_4",
    "weekly_cycle_sin", "weekly_cycle_cos", "monthly_cycle_sin", "monthly_cycle_cos",
    "ema_3", "ema_7", "ema_14",
    "z_score_4", "z_score_8",
    "lag1_x_month", "rolling_mean_4_x_trend"
]

# Sheet analysis threshold - Lowered for better ML coverage
SHEET_ML_THRESHOLD = 100

# ============================================================================
# Column Mapping Configuration
# ============================================================================

# Column mapping dictionaries for Turkish/English compatibility
DATE_COLS = [
    "tarih", "date", "Tarih", "Date", "DATE", "TARİH",
    "invoice_date", "InvoiceDate", "fatura_tarihi", "siparis_tarihi"
]

STOCK_COLS = [
    "stok_kodu", "stock_code", "stockcode", "StockCode", "STOCK_CODE",
    "product_code", "ProductCode", "ürün_kodu", "urun_kodu"
]

PRODUCT_NAME_COLS = [
    "ürün_adı", "urun_adi", "product_name", "ProductName", "Description",
    "ürün_tanımı", "urun_tanimi", "açıklama", "aciklama", "Urun", "urun"
]

QTY_COLS = [
    "miktar", "quantity", "Quantity", "QUANTITY", "Miktar", "MİKTAR",
    "adet", "Adet", "qty", "QTY", "İhtiyaç Kg", "ihtiyac_kg", "ihtiyaç_kg", 
    "need_kg", "requirement_kg", "kg", "KG", "Satis_Miktari", "satis_miktari"
]

PRICE_COLS = [
    "birim_fiyat", "unit_price", "UnitPrice", "Price", "fiyat", "Fiyat",
    "birim_fiyati", "unitprice", "Fiyat"
]

CUSTOMER_COLS = [
    "müşteri_kodu", "musteri_kodu", "customer_id", "CustomerID", "CUSTOMER_ID",
    "müşteri", "musteri", "customer", "Customer"
]

# ============================================================================
# Library Availability Checks
# ============================================================================

# Check for optional libraries
def check_library_availability():
    """Check availability of optional libraries"""
    availability = {}
    
    # Seaborn
    try:
        import seaborn as sns  # type: ignore
        availability['seaborn'] = True
    except ImportError:
        availability['seaborn'] = False
    
    # LightGBM
    try:
        from lightgbm import LGBMRegressor  # type: ignore
        availability['lightgbm'] = True
    except ImportError:
        availability['lightgbm'] = False
    
    # XGBoost
    try:
        import xgboost as xgb  # type: ignore
        availability['xgboost'] = True
    except ImportError:
        availability['xgboost'] = False
    
    # CatBoost
    try:
        import catboost as cb  # type: ignore
        availability['catboost'] = True
    except ImportError:
        availability['catboost'] = False
    
    # Prophet
    try:
        from prophet import Prophet  # type: ignore
        availability['prophet'] = True
    except ImportError:
        availability['prophet'] = False
        
    return availability

# Get library availability
LIBRARY_AVAILABILITY = check_library_availability()
