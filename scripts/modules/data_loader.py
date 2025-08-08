#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data loading module for the sales forecasting system.
Contains functions for loading and preprocessing data from various formats.
"""

import sqlite3
from pathlib import Path
from typing import Dict
import pandas as pd
from .utils.metrics import log_output
from .config import (
    DATE_COLS, STOCK_COLS, PRODUCT_NAME_COLS, QTY_COLS, 
    PRICE_COLS, CUSTOMER_COLS, TARGET_COL
)

# ============================================================================
# Column Mapping
# ============================================================================

def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Turkish/English column names to standard format.
    """
    log_output(f"📊 Column mapping in progress... Current columns: {list(df.columns)}")
    
    # Handle empty DataFrame
    if len(df) == 0 or len(df.columns) == 0:
        log_output("⚠️  Warning: Empty dataset provided")
        return pd.DataFrame(columns=["InvoiceDate", "StockCode", "Quantity", "UnitPrice", "CustomerID"])
    
    # Create a copy for output
    df_out = pd.DataFrame()
    
    # Find matching columns
    col_date = next((col for col in df.columns if col in DATE_COLS), None)
    col_stock = next((col for col in df.columns if col in STOCK_COLS), None)
    col_product_name = next((col for col in df.columns if col in PRODUCT_NAME_COLS), None)
    col_qty = next((col for col in df.columns if col in QTY_COLS), None)
    col_price = next((col for col in df.columns if col in PRICE_COLS), None)
    col_customer = next((col for col in df.columns if col in CUSTOMER_COLS), None)

    # Map Date column
    if col_date:
        df_out["InvoiceDate"] = pd.to_datetime(df[col_date], errors='coerce')
        log_output(f"✅ Date column found: {col_date}")
    else:
        log_output("❌ Date column not found! Using first column as date.")
        df_out["InvoiceDate"] = pd.to_datetime(df.iloc[:, 0], errors='coerce')

    # Map StockCode
    if col_stock:
        df_out["StockCode"] = df[col_stock].astype(str)
        log_output(f"✅ Stock code column found: {col_stock}")
    elif col_product_name:
        df_out["StockCode"] = df[col_product_name].astype(str)
        log_output(f"✅ Product name used as stock code: {col_product_name}")
    else:
        df_out["StockCode"] = "UNKNOWN_PRODUCT"
        log_output("⚠️  Warning: Stock code not found. Set as UNKNOWN_PRODUCT.")

    # Map Quantity
    if col_qty:
        df_out["Quantity"] = pd.to_numeric(df[col_qty], errors='coerce').fillna(1)
        log_output(f"✅ Quantity column found: {col_qty}")
    else:
        df_out["Quantity"] = 1
        log_output("⚠️  Warning: Quantity column not found. Set as 1.")

    # Map UnitPrice
    if col_price:
        df_out["UnitPrice"] = pd.to_numeric(df[col_price], errors='coerce').fillna(10)
        log_output(f"✅ Price column found: {col_price}")
    else:
        df_out["UnitPrice"] = 10  # Default price
        log_output("⚠️  Warning: Price column not found. Set as 10.")

    # Map CustomerID
    if col_customer:
        df_out["CustomerID"] = df[col_customer].fillna("UNKNOWN")
        log_output(f"✅ Customer column found: {col_customer}")
    else:
        df_out["CustomerID"] = "DEFAULT_CUSTOMER"
        log_output("⚠️  Warning: Customer ID not found. Set as DEFAULT_CUSTOMER.")

    log_output(f"📊 Column mapping completed. Row count: {len(df_out)}")
    return df_out

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_excel_sheets_separately(path: Path) -> Dict[str, pd.DataFrame]:
    """
    Load each Excel sheet separately and return as dictionary.
    """
    log_output(f"📂 Loading Excel file by sheets: {path}")
    
    try:
        excel_data = pd.read_excel(path, sheet_name=None)
        sheets_data = {}
        
        for sheet_name, sheet_df in excel_data.items():
            if len(sheet_df) > 0:  # Only process non-empty sheets
                log_output(f"📄 Sheet '{sheet_name}': {len(sheet_df)} rows")
                sheets_data[sheet_name] = sheet_df
            else:
                log_output(f"⚠️  Sheet '{sheet_name}' is empty, skipping")
                
        log_output(f"✅ Total {len(sheets_data)} sheets loaded")
        return sheets_data
        
    except Exception as e:
        log_output(f"❌ Error loading Excel file: {e}")
        raise

def load_dataset_generic(path: Path) -> Dict[str, pd.DataFrame]:
    """
    Load dataset from various formats, return as sheet dictionary.
    """
    ext = path.suffix.lower()
    log_output(f"🔍 File format detected: {path} (ext={ext})")

    if ext in [".xlsx", ".xls"]:
        return load_excel_sheets_separately(path)
        
    elif ext == ".csv":
        df = pd.read_csv(path)
        log_output(f"📄 CSV file loaded: {len(df)} rows")
        
        # Check if we have product-based data (multiple products in one CSV)
        if 'Urun' in df.columns or 'urun' in df.columns or 'Product' in df.columns:
            # Split by product/urun column
            product_col = None
            for col in ['Urun', 'urun', 'Product', 'product', 'PRODUCT']:
                if col in df.columns:
                    product_col = col
                    break
            
            if product_col:
                sheets_data = {}
                for product in df[product_col].unique():
                    product_df = df[df[product_col] == product].copy()
                    sheets_data[str(product)] = product_df
                    log_output(f"📄 Product '{product}': {len(product_df)} rows")
                log_output(f"✅ CSV split into {len(sheets_data)} product sheets")
                return sheets_data
        
        # Fallback to single sheet
        return {"Sheet1": df}
        
    elif ext in [".sqlite", ".db"]:
        conn = sqlite3.connect(path)
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        sheets_data = {}
        for table_name in tables['name']:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            sheets_data[table_name] = df
            log_output(f"📄 Table '{table_name}': {len(df)} rows")
        conn.close()
        return sheets_data
        
    elif ext == ".sql":
        conn = sqlite3.connect(":memory:")
        with open(path, 'r', encoding='utf-8') as f:
            sql_script = f.read()
        conn.executescript(sql_script)
        
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        sheets_data = {}
        for table_name in tables['name']:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            sheets_data[table_name] = df
            log_output(f"📄 Table '{table_name}': {len(df)} rows")
        conn.close()
        return sheets_data
        
    else:
        raise ValueError(f"Unsupported file format: {ext}")

# ============================================================================
# Data Preparation
# ============================================================================

def prepare_weekly_data(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """
    Prepare weekly aggregated data for modeling.
    """
    log_output(f"📊 {sheet_name}: Preparing weekly data...")
    
    # Drop rows with invalid dates
    df = df.dropna(subset=["InvoiceDate"])
    
    # Create total sales and quantity metrics
    df["TotalSales"] = df["Quantity"] * df["UnitPrice"]
    
    # Create week column
    df["week"] = df["InvoiceDate"].dt.to_period("W").dt.start_time
    
    # Group by week and stock code
    weekly = df.groupby(["week", "StockCode"]).agg({
        "TotalSales": "sum",
        "Quantity": "sum"
    }).reset_index()
    
    # Aggregate all products
    weekly_total = weekly.groupby("week").agg({
        "TotalSales": "sum",
        "Quantity": "sum"
    }).reset_index()
    
    # Use Quantity (adet) instead of TotalSales (TL) for forecasting
    weekly_total = weekly_total.rename(columns={"Quantity": TARGET_COL})
    
    log_output(f"📊 {sheet_name}: {len(weekly_total)} weekly data prepared (using quantity in adet)")
    return weekly_total
