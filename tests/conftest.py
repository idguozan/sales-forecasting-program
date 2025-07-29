"""
Pytest configuration and shared fixtures
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture
def sample_sales_data():
    """Sample sales data for testing with InvoiceDate column"""
    dates = pd.date_range('2023-01-01', periods=52, freq='W')
    data = {
        'InvoiceDate': dates,  # prepare_weekly_data fonksiyonu bu kolonu bekliyor
        'İhtiyaç Kg': np.random.randint(100, 1000, 52),
        'StockCode': ['TEST001'] * 52,  # StockCode kolonu
        'Quantity': np.random.randint(1, 10, 52),  # Quantity kolonu ekle
        'UnitPrice': np.random.uniform(5, 50, 52)  # UnitPrice kolonu ekle
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_large_dataset():
    """Large dataset for ML testing (>100 rows)"""
    dates = pd.date_range('2023-01-01', periods=120, freq='W')
    data = {
        'week': dates,
        'İhtiyaç Kg': np.random.randint(50, 500, 120),
        'Stok Kodu': ['TEST_LARGE'] * 120
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_small_dataset():
    """Small dataset for simple forecasting (<100 rows)"""
    dates = pd.date_range('2023-01-01', periods=20, freq='W')
    data = {
        'week': dates,
        'İhtiyaç Kg': np.random.randint(50, 200, 20),
        'Stok Kodu': ['TEST_SMALL'] * 20
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_excel_file():
    """Create temporary Excel file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        dates = pd.date_range('2023-01-01', periods=20, freq='W')
        data = pd.DataFrame({
            'week': dates,
            'İhtiyaç Kg': np.random.randint(50, 200, 20),
            'Stok Kodu': ['TEST001'] * 20
        })
        data.to_excel(tmp.name, sheet_name='Test_Sheet', index=False)
        yield tmp.name
        # Cleanup
        os.unlink(tmp.name)

@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        'FORECAST_HORIZON_WEEKS': 12,
        'SHEET_ML_THRESHOLD': 100,
        'TARGET_COL': 'İhtiyaç Kg',
        'STOCK_CODE_COL': 'Stok Kodu',
        'WEEK_COL': 'week'
    }

@pytest.fixture
def sample_mapped_data():
    """Sample data with proper column mapping (total_sales column)"""
    dates = pd.date_range('2023-01-01', periods=30, freq='W')
    return pd.DataFrame({
        'week': dates,
        'total_sales': np.random.randint(100, 500, 30),  # CONFIG'de TARGET_COL = "total_sales"
        'quantity': np.random.randint(10, 50, 30),
        'stock_code': ['MAPPED_TEST'] * 30,
        'price': np.random.uniform(10, 100, 30)
    })

@pytest.fixture
def sample_csv_file():
    """Create temporary CSV file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        dates = pd.date_range('2023-01-01', periods=20, freq='W')
        data = pd.DataFrame({
            'week': dates,
            'İhtiyaç Kg': np.random.randint(50, 200, 20),
            'Stok Kodu': ['TEST001'] * 20
        })
        data.to_csv(tmp.name, index=False)
        yield tmp.name
        # Cleanup
        os.unlink(tmp.name)
