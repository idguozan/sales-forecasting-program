"""
Unit tests for configuration module
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.modules.config import (
    FORECAST_HORIZON_WEEKS,
    SHEET_ML_THRESHOLD,
    TARGET_COL,
    STOCK_COLS,
    PROJECT_ROOT,
    PLOTS_DIR,
    REPORTS_DIR,
    UPLOADS_DIR
)

def test_config_constants_exist():
    """Test that all required configuration constants exist"""
    # Check that constants are defined
    assert isinstance(FORECAST_HORIZON_WEEKS, int)
    assert isinstance(SHEET_ML_THRESHOLD, int)
    assert isinstance(TARGET_COL, str)
    assert isinstance(STOCK_COLS, list)
    
    # Check reasonable values
    assert FORECAST_HORIZON_WEEKS > 0
    assert SHEET_ML_THRESHOLD > 0
    assert len(TARGET_COL) > 0
    assert len(STOCK_COLS) > 0

def test_directory_paths_exist():
    """Test that directory paths are valid Path objects"""
    assert isinstance(PROJECT_ROOT, Path)
    assert isinstance(PLOTS_DIR, Path)
    assert isinstance(REPORTS_DIR, Path)
    assert isinstance(UPLOADS_DIR, Path)
    
    # Check that directories exist (they should be created on import)
    assert PLOTS_DIR.exists()
    assert REPORTS_DIR.exists()
    assert UPLOADS_DIR.exists()

def test_forecast_horizon_reasonable():
    """Test that forecast horizon is reasonable (between 1-52 weeks)"""
    assert 1 <= FORECAST_HORIZON_WEEKS <= 52

def test_ml_threshold_reasonable():
    """Test that ML threshold is reasonable"""
    assert SHEET_ML_THRESHOLD >= 1

def test_column_names_not_empty():
    """Test that column names are not empty strings"""
    assert TARGET_COL.strip() != ""
    assert len(STOCK_COLS) > 0
    assert all(col.strip() != "" for col in STOCK_COLS)

def test_project_structure():
    """Test that expected project structure exists"""
    # Check that key directories exist relative to PROJECT_ROOT
    expected_dirs = ["scripts", "app", "uploads"]
    for dir_name in expected_dirs:
        expected_path = PROJECT_ROOT / dir_name
        assert expected_path.exists(), f"Expected directory {dir_name} not found"
