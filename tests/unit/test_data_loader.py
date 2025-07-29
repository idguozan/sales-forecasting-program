"""
Unit tests for data loading module
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.modules.data_loader import (
    map_columns,
    load_dataset_generic
)

def test_column_mapping_basic():
    """Test basic column mapping functionality"""
    # Create a simple DataFrame to test mapping
    test_df = pd.DataFrame({
        'hafta': [1, 2, 3],
        'miktar': [100, 200, 300],
        'urun_kodu': ['A', 'B', 'C']
    })
    
    result = map_columns(test_df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3

def test_map_columns_output_structure(sample_sales_data):
    """Test that map_columns produces expected output structure"""
    result = map_columns(sample_sales_data)
    
    # Check required columns exist (these are the actual output columns)
    expected_cols = ['InvoiceDate', 'StockCode', 'Quantity', 'UnitPrice', 'CustomerID']
    for col in expected_cols:
        assert col in result.columns, f"Missing required column: {col}"
    
    # Check data integrity
    assert len(result) == len(sample_sales_data)
    assert result['Quantity'].notna().all()

def test_load_dataset_generic_csv(sample_csv_file):
    """Test loading CSV file"""
    from pathlib import Path
    result = load_dataset_generic(Path(sample_csv_file))  # Path nesnesine çevir
    
    assert isinstance(result, dict)
    assert len(result) > 0
    
    # Should have one sheet named 'Sheet1' for CSV files
    assert 'Sheet1' in result
    df = result['Sheet1']
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

def test_load_dataset_generic_excel(sample_excel_file):
    """Test loading Excel file with multiple sheets"""
    from pathlib import Path
    result = load_dataset_generic(Path(sample_excel_file))  # Path nesnesine çevir
    
    assert isinstance(result, dict)
    assert len(result) > 0
    
    # Check that all values are DataFrames
    for sheet_name, df in result.items():
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
