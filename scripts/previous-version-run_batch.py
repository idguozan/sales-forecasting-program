#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Batch sales forecasting pipeline.

Usage (optional argument):
    python scripts/run_batch.py path/to/datafile.xlsx

If no argument is given:
    uploads/last_uploaded.json is read and the most recently uploaded dataset is used.

Supported data formats:
    - .xlsx / .xls (Excel)
    - .csv
    - .sql        (SQLite script)
    - .sqlite / .db (SQLite database file)

Outputs:
    app/static/plots/   -> product-level PNG charts
    app/static/reports/ -> forecast_report.pdf, total_weekly_forecast.png, prophet_future_forecast.csv
    project_root/       -> future_forecast.csv (RF predictions; legacy code compatibility)

Note: Prophet is optional. If not installed, RF predictions will run, Prophet will be skipped.
"""

import os
import sys
import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
# LightGBM 
try:
    from lightgbm import LGBMRegressor  # type: ignore
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

# Prophet 
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:  # It's okay if Prophet is missing.
    PROPHET_AVAILABLE = False


# ============================================================================
# Konfig
# ============================================================================


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Application static directories (served by FastAPI)
APP_STATIC_DIR = PROJECT_ROOT / "app" / "static"
PLOTS_DIR = APP_STATIC_DIR / "plots"
REPORTS_DIR = APP_STATIC_DIR / "reports"

# Uploads folder (populated by FastAPI upload router)
UPLOADS_DIR = PROJECT_ROOT / "uploads"
LAST_UPLOADED_JSON = UPLOADS_DIR / "last_uploaded.json"

# Output files
FUTURE_CSV_ROOT = PROJECT_ROOT / "future_forecast.csv"  # RF prediction (legacy)
PROPHET_CSV = REPORTS_DIR / "prophet_future_forecast.csv"
PDF_REPORT = REPORTS_DIR / "forecast_report.pdf"
TOTAL_PLOT_PNG = REPORTS_DIR / "total_weekly_forecast.png"

# Forecast parameters
# AGG_FREQ: "W" = weekly, "M" = monthly (month start)
AGG_FREQ = "M"  # You can change this for experiments ("M")

# Forecast horizon (number of periods):
#   AGG_FREQ = "W" → period = week
#   AGG_FREQ = "M" → period = month
FORECAST_HORIZON_PERIODS = 6   # Example: 12 weeks or 12 months

# Backward compatibility warning for legacy code
FORECAST_HORIZON_WEEKS = FORECAST_HORIZON_PERIODS
MAX_PRODUCTS: Optional[int] = None  # None = unlimited
MIN_WEEKS_REQUIRED = 0        # Eligible product filter


# ============================================================================
# Helper: Read the most recently uploaded data file
# ============================================================================
def get_last_uploaded_path() -> Optional[Path]:
    if not LAST_UPLOADED_JSON.exists():
        return None
    try:
        with LAST_UPLOADED_JSON.open("r", encoding="utf-8") as f:
            data = json.load(f)
        p = data.get("last_file")
        if not p:
            return None
        p = Path(p).expanduser()
        if not p.is_absolute():
            p = (UPLOADS_DIR / p).resolve()
        return p if p.exists() else None
    except Exception:
        return None


# ============================================================================
# Helper: Column mapping (flexible)
# ============================================================================
REQUIRED_CANONICAL = ["InvoiceDate", "StockCode", "Quantity", "UnitPrice", "CustomerID"]
COLUMN_SYNONYMS = {
    "InvoiceDate": ["invoicedate", "invoice_date", "date", "datetime", "timestamp", "orderdate", "order_date"],
    "StockCode": ["stockcode", "stock_code", "sku", "productcode", "product_code", "itemcode", "item_code"],
    "Quantity": ["quantity", "qty", "units", "amount", "sales_units"],
    "UnitPrice": ["unitprice", "price", "unit_price", "sales_price", "netprice"],
    "CustomerID": ["customerid", "customer_id", "custid", "clientid", "accountid", "buyerid"],
}


## ===============================
# Column Mapping (Automatic, Turkish + English)
## ===============================
def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts input DataFrame to expected column names:

        InvoiceDate, StockCode, Quantity, UnitPrice, CustomerID

    Column names may be Turkish/English/misspelled. If not found,
    the most logical fallback is created (e.g. generate fake StockCode from ProductName, set CustomerID to 0, etc.)

    Returns: *copy* DataFrame with new columns.
    """
    df_in = df.copy()

    # --- Convert columns to string & normalize ---
    # (lowercase, trim, space->underscore, Turkish character transliteration)
    def _norm(s: str) -> str:
        s = str(s).strip().lower().replace(" ", "_")
        repl = {
            "ş": "s", "ı": "i", "İ": "i", "ğ": "g", "ü": "u",
            "ö": "o", "ç": "c", "â": "a", "ê": "e", "î": "i", "ô": "o", "û": "u"
        }
        out = []
        for ch in s:
            out.append(repl.get(ch, ch))
        return "".join(out)

    norm_map = {_norm(c): c for c in df_in.columns}  # norm -> orijinal kolon
    norm_cols = list(norm_map.keys())

    def _pick(synonyms):
        """Returns the first matching normalized name from synonym list, or None."""
        for cand in synonyms:
            cand_norm = _norm(cand)
            # direct match?
            if cand_norm in norm_map:
                return norm_map[cand_norm]
            # sometimes column name may contain a longer phrase (contains match)
            for nc in norm_cols:
                if cand_norm in nc:
                    return norm_map[nc]
        return None

    # --- Synonym lists ---
    syn = {
        "InvoiceDate": [
            "invoicedate", "invoice_date", "date", "orderdate", "order_date",
            "tarih", "satis_tarihi", "satis_tarih", "satış_tarihi", "satistarih"
        ],
        "StockCode": [
            "stockcode", "stock_code", "sku", "urun_kodu", "ürün_kodu",
            "urunkodu", "urun_kod", "product_id", "productcode", "ürün", "urun",
            "siparis_id", "siparisno", "order_id"
        ],
        "Quantity": [
            "quantity", "qty", "satıs_adedi", "satis_adedi", "satış_adedi",
            "adet", "miktar", "units", "units_sold", "sales_qty", "satilan_adet",
            "sales_quantity", "qty_sold", "satis_miktari", "ihtiyac", "ihtiyaç",
            "ihtiyac_kg", "ihtiyaç_kg"
        ],
        "UnitPrice": [
            "unitprice", "unit_price", "price", "birim_fiyat", "fiyat",
            "unit_cost", "cost", "satis_fiyati", "satis_fiyat"
        ],
        "CustomerID": [
            "customerid", "customer_id", "musteri_id", "müşteri_id",
            "musteri", "müşteri", "client_id", "account_id", "cust_id"
        ],
        # Note: Product name (to be used if StockCode is missing)
        "ProductName": [
            "description", "urun_adi", "ürün_adı", "product_name", "product",
            "item", "item_name", "urunisim", "urun", "ürün"
        ],
    }

    # --- Find original name for each target column ---
    col_invoice = _pick(syn["InvoiceDate"])
    col_stock   = _pick(syn["StockCode"])
    col_qty     = _pick(syn["Quantity"])
    col_price   = _pick(syn["UnitPrice"])
    col_cust    = _pick(syn["CustomerID"])
    col_pname   = _pick(syn["ProductName"])

    # --- Fallbacks ---
    # If StockCode is missing -> generate code from ProductName
    if col_stock is None:
        if col_pname is not None:
            print("[!] StockCode missing, generating from ProductName...")
            df_in["_tmp_stockcode"] = (
                df_in[col_pname].astype(str).str.slice(0, 20).str.upper()
            )
            col_stock = "_tmp_stockcode"
        else:
            print("[!] StockCode missing, using row index...")
            df_in["_tmp_stockcode"] = df_in.index.astype(str)
            col_stock = "_tmp_stockcode"

    # If Quantity is missing -> assume 1
    if col_qty is None:
        print("[!] Quantity missing, assuming 1...")
        df_in["_tmp_qty"] = 1
        col_qty = "_tmp_qty"

    # If UnitPrice is missing -> assume 1
    if col_price is None:
        print("[!] UnitPrice missing, assuming 1.0...")
        df_in["_tmp_price"] = 1.0
        col_price = "_tmp_price"

    # If CustomerID is missing -> assume 0
    if col_cust is None:
        print("[!] CustomerID missing, assuming 0...")
        df_in["_tmp_cust"] = 0
        col_cust = "_tmp_cust"

    # If InvoiceDate is missing -> error (cannot build time series without date)
    if col_invoice is None:
        raise ValueError("InvoiceDate / Date column could not be detected (required).")

    # --- Output DF ---
    out = df_in[[col_invoice, col_stock, col_qty, col_price, col_cust]].copy()
    out.columns = ["InvoiceDate", "StockCode", "Quantity", "UnitPrice", "CustomerID"]
    return out



# =============================================================================
# Helper: Load data (multiple formats)
# =============================================================================
def load_dataset_generic(path: Path) -> pd.DataFrame:
    import pandas as pd
    import sqlite3

    ext = path.suffix.lower()
    print(f"[*] Data file detected: {path} (ext={ext})")

    if ext in [".xlsx", ".xls"]:
        # If Excel file has multiple sheets, combine all
        excel_data = pd.read_excel(path, sheet_name=None)
        if isinstance(excel_data, dict):  # Multiple sheets
            print(f"[*] {len(excel_data)} sheets detected in Excel file, combining...")
            df = pd.concat(excel_data.values(), ignore_index=True)
        else:  # Single sheet
            df = excel_data
    elif ext == ".csv":
        df = pd.read_csv(path)
    elif ext in [".sqlite", ".db"]:
        print("[*] Reading SQLite database...")
        conn = sqlite3.connect(path)
        tbls = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        if tbls.empty:
            conn.close()
            raise ValueError("No table found in SQLite file.")
        table_name = tbls["name"].iloc[0]
        print(f"    -> Table to read: {table_name}")
        df = pd.read_sql_query(f"SELECT * FROM {table_name};", conn)
        conn.close()
    elif ext == ".sql":
        print("[*] Running SQL script (in-memory SQLite)...")
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            sql_script = f.read()
        conn = sqlite3.connect(":memory:")
        conn.executescript(sql_script)
        tbls = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        if tbls.empty:
            conn.close()
            raise ValueError(".sql script ran but no table was created.")
        table_name = tbls["name"].iloc[0]
        print(f"    -> Table to read: {table_name}")
        df = pd.read_sql_query(f"SELECT * FROM {table_name};", conn)
        conn.close()
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # ---- Normalize column names ----
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    print(f"[*] Columns normalized: {df.columns.tolist()}")
    print(f"[*] Data loaded. Row count: {len(df)}")
    return df


# ============================================================================
# Prophet (forecast for a single product)
# ============================================================================
def prophet_forecast_single(product_df: pd.DataFrame, weeks: int):
    """
    product_df: kolonları ['week','qty']
    """
    if not PROPHET_AVAILABLE or product_df.empty:
        return None
    df_prop = product_df[["week", "qty"]].rename(columns={"week": "ds", "qty": "y"}).copy()
    model = Prophet()
    model.fit(df_prop)
    future = model.make_future_dataframe(periods=weeks, freq="W")
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def simple_forecast(df, forecast_horizon=12):
    """
    Simple forecast for small datasets: repeats the last observed value.
    """
    if len(df) == 0:
        return pd.DataFrame(columns=["week", "predicted_qty"])
    last_qty = df["Quantity"].iloc[-1]
    last_date = pd.to_datetime(df["InvoiceDate"].max())
    future_weeks = [last_date + pd.Timedelta(weeks=i+1) for i in range(forecast_horizon)]
    return pd.DataFrame({
        "week": future_weeks,
        "predicted_qty": [last_qty] * forecast_horizon
    })


def ml_forecast_pipeline(df, forecast_horizon=12):
    """
    Full ML pipeline for a single product: feature engineering, model training, rolling forecast.
    Returns forecast DataFrame and metrics dict.
    """
    # Clean and preprocess
    df = df.dropna(subset=["CustomerID"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    df["week"] = df["InvoiceDate"].dt.to_period("W").dt.start_time
    df["line_total"] = df["Quantity"] * df["UnitPrice"]
    weekly = (
        df.groupby(["week"])
        .agg(
            qty=("Quantity", "sum"),
            avg_price=("UnitPrice", "mean"),
            revenue=("line_total", "sum"),
        )
        .reset_index()
        .sort_values(["week"])
        .reset_index(drop=True)
    )
    # Feature engineering
    weekly["month"] = weekly["week"].dt.month
    weekly["year"] = weekly["week"].dt.year
    weekly["week_no"] = weekly["week"].dt.isocalendar().week.astype(int)
    for lag in [1, 4, 8]:
        weekly[f"qty_lag_{lag}"] = weekly["qty"].shift(lag)
    for window in [4, 8]:
        weekly[f"roll_mean_{window}"] = weekly["qty"].shift(1).rolling(window).mean()
        weekly[f"roll_std_{window}"] = weekly["qty"].shift(1).rolling(window).std()
    year_lag = 52
    weekly[f"qty_lag_{year_lag}"] = weekly["qty"].shift(year_lag)
    weekly["price_diff"] = weekly["avg_price"].diff()
    weekly["promo_flag"] = (weekly["price_diff"] < 0).astype(int)
    weekly_model = weekly.dropna().reset_index(drop=True)
    FEATURE_COLS = [
        "avg_price", "month", "week_no", "qty_lag_1", "qty_lag_4", "qty_lag_8", f"qty_lag_{year_lag}",
        "roll_mean_4", "roll_std_4", "roll_mean_8", "roll_std_8", "promo_flag"
    ]
    TARGET_COL = "qty"
    # Train/test split
    if len(weekly_model) < 20:
        # Not enough data for ML
        return simple_forecast(df, forecast_horizon), None
    max_week = weekly_model["week"].max()
    cutoff_week = max_week - pd.Timedelta(weeks=forecast_horizon)
    train = weekly_model[weekly_model["week"] <= cutoff_week].copy()
    test = weekly_model[weekly_model["week"] > cutoff_week].copy()
    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_test = test[FEATURE_COLS]
    y_test = test[TARGET_COL]
    # Model training
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    # Backtest metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred) if len(y_test) > 0 else None
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) if len(y_test) > 0 else None
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100 if len(y_test) > 0 else None
    wape = np.sum(np.abs(y_test - y_pred)) / np.sum(y_test) * 100 if len(y_test) > 0 else None
    metrics = {"mae": mae, "rmse": rmse, "mape": mape, "wape": wape}
    # Rolling forecast
    qty_hist = weekly["qty"].to_list()
    last_price = weekly["avg_price"].iloc[-1]
    last_week = weekly["week"].max()
    future_qtys = []
    for _ in range(forecast_horizon):
        next_week = last_week + pd.Timedelta(weeks=1)
        history = qty_hist + future_qtys
        qty_lag_1 = history[-1]
        qty_lag_4 = history[-4] if len(history) >= 4 else qty_lag_1
        qty_lag_8 = history[-8] if len(history) >= 8 else qty_lag_4
        roll_mean_4 = np.mean(history[-4:]) if len(history) >= 4 else np.mean(history)
        roll_std_4 = np.std(history[-4:]) if len(history) >= 4 else 0.0
        roll_mean_8 = np.mean(history[-8:]) if len(history) >= 8 else roll_mean_4
        roll_std_8 = np.std(history[-8:]) if len(history) >= 8 else roll_std_4
        qty_lag_year = history[-year_lag] if len(history) >= year_lag else qty_lag_1
        month = next_week.month
        week_no = next_week.isocalendar().week
        promo_flag = 0
        X_future = pd.DataFrame({
            "avg_price": [last_price],
            "month": [month],
            "week_no": [week_no],
            "qty_lag_1": [qty_lag_1],
            "qty_lag_4": [qty_lag_4],
            "qty_lag_8": [qty_lag_8],
            f"qty_lag_{year_lag}": [qty_lag_year],
            "roll_mean_4": [roll_mean_4],
            "roll_std_4": [roll_std_4],
            "roll_mean_8": [roll_mean_8],
            "roll_std_8": [roll_std_8],
            "promo_flag": [promo_flag],
        })
        pred_qty = model.predict(X_future)[0]
        pred_qty = max(pred_qty, 0)
        future_qtys.append(pred_qty)
        last_week = next_week
    future_weeks = [weekly["week"].max() + pd.Timedelta(weeks=i+1) for i in range(forecast_horizon)]
    forecast_df = pd.DataFrame({
        "week": future_weeks,
        "predicted_qty": future_qtys
    })
    return forecast_df, metrics


def process_all_sheets(path, forecast_horizon=12):
    """
    Process each sheet (product) independently: use simple or ML forecast based on row count.
    """
    excel_data = pd.read_excel(path, sheet_name=None)
    results = []
    for sheet_name, df in excel_data.items():
        if len(df) < 10:
            continue  # skip sheets with too little data
        if len(df) < 150:
            forecast_df = simple_forecast(df, forecast_horizon)
            model_type = "Simple"
            metrics = None
        else:
            forecast_df, metrics = ml_forecast_pipeline(df, forecast_horizon)
            model_type = "ML"
        results.append({
            "sheet": sheet_name,
            "forecast": forecast_df,
            "model_type": model_type,
            "metrics": metrics
        })
    return results


def generate_pdf_report(results, output_path):
    """
    Generate a combined PDF report with a section for each product (sheet).
    """
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    with PdfPages(output_path) as pdf:
        for res in results:
            sheet = res["sheet"]
            forecast = res["forecast"]
            model_type = res["model_type"]
            metrics = res["metrics"]
            plt.figure(figsize=(10, 6))
            plt.title(f"Forecast for {sheet} ({model_type})")
            plt.plot(forecast["week"], forecast["predicted_qty"], label="Forecast", marker="o")
            plt.xlabel("Week")
            plt.ylabel("Predicted Quantity")
            plt.legend()
            plt.tight_layout()
            pdf.savefig(); plt.close()
            # Metrics page (if ML)
            if model_type == "ML" and metrics is not None:
                plt.figure(figsize=(8, 4))
                plt.axis("off")
                metrics_text = (f"MAE: {metrics['mae']:.2f}\nRMSE: {metrics['rmse']:.2f}\nMAPE: {metrics['mape']:.2f}%\nWAPE: {metrics['wape']:.2f}%")
                plt.text(0.1, 0.8, metrics_text, fontsize=12, family="monospace")
                plt.title(f"Metrics for {sheet}")
                pdf.savefig(); plt.close()

# Main batch run (replace previous main)
def main():
    # 1. Data file selection
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1]).expanduser().resolve()
    else:
        data_path = get_last_uploaded_path()
    if not data_path or not data_path.exists():
        raise FileNotFoundError("No dataset found. Please upload data from the admin panel first.")
    print(f"[*] Dataset to be used: {data_path}")
    # 2. Process all sheets independently
    results = process_all_sheets(data_path, forecast_horizon=12)
    # 3. Generate combined PDF report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_pdf = REPORTS_DIR / "forecast_report.pdf"
    generate_pdf_report(results, output_pdf)
    print(f"PDF report generated: {output_pdf}")

if __name__ == "__main__":
    main()
