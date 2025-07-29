"""
Unit tests for data_loader module
"""
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import os

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.modules.data_loader import (
    load_dataset_generic,
    map_columns,
    normalize_column_name
)

class TestDataLoader:
    
    def test_normalize_column_name(self):
        """Test column name normalization"""
        test_cases = [
            ("İhtiyaç Kg", "ihtiyac_kg"),
            ("Stock Code", "stock_code"),
            ("INVOICE DATE", "invoice_date"),
            ("  WeEk   ", "week"),
            ("Müşteri_ID", "musteri_id"),
            ("Stok Kodu", "stok_kodu"),
            ("Satış Tarihi", "satis_tarihi")
        ]
        
        for input_col, expected in test_cases:
            result = normalize_column_name(input_col)
            assert result == expected, f"Input: {input_col}, Expected: {expected}, Got: {result}"
    
    def test_map_columns_success(self, sample_sales_data):
        """Test successful column mapping with Turkish columns"""
        # Rename columns to Turkish variants
        turkish_data = sample_sales_data.rename(columns={
            'week': 'hafta',
            'İhtiyaç Kg': 'miktar',
            'Stok Kodu': 'urun_kodu'
        })
        
        result = map_columns(turkish_data)
        
        # Check required columns exist
        required_cols = ['week', 'quantity', 'stock_code']
        for col in required_cols:
            assert col in result.columns, f"Missing required column: {col}"
        
        # Check data integrity
        assert len(result) == len(turkish_data)
        assert not result.empty
    
    def test_map_columns_missing_data(self):
        """Test column mapping with missing columns - should create defaults"""
        df = pd.DataFrame({
            'tarih': pd.date_range('2023-01-01', periods=5),
            'miktar': [100, 200, 150, 300, 250]
        })
        
        result = map_columns(df)
        
        # Should still have required columns (with defaults)
        expected_cols = ['week', 'quantity', 'stock_code']
        for col in expected_cols:
            assert col in result.columns, f"Missing required column: {col}"
        
        # Check default values were created
        assert result['stock_code'].notna().all()
    
    def test_map_columns_edge_cases(self):
        """Test column mapping edge cases"""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        result = map_columns(empty_df)
        assert isinstance(result, pd.DataFrame)
        
        # Single row
        single_row = pd.DataFrame({
            'week': [pd.Timestamp('2023-01-01')],
            'İhtiyaç Kg': [100],
            'Stok Kodu': ['TEST']
        })
        result = map_columns(single_row)
        assert len(result) == 1
        assert 'quantity' in result.columns
    
    def test_load_dataset_generic_excel(self, sample_excel_file):
        """Test loading Excel file"""
        result = load_dataset_generic(Path(sample_excel_file))
        
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Check sheet data
        for sheet_name, df in result.items():
            assert isinstance(df, pd.DataFrame)
            assert not df.empty
    
    def test_load_dataset_generic_nonexistent_file(self):
        """Test loading non-existent file"""
        result = load_dataset_generic(Path("nonexistent_file_12345.xlsx"))
        assert result == {}
    
    def test_load_dataset_generic_csv(self):
        """Test loading CSV file"""
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp.write("week,quantity,stock_code\n")
            tmp.write("2023-01-01,100,TEST001\n")
            tmp.write("2023-01-08,150,TEST001\n")
            tmp_path = tmp.name
        
        try:
            result = load_dataset_generic(Path(tmp_path))
            assert isinstance(result, dict)
            assert len(result) > 0
        finally:
            os.unlink(tmp_path)
    
    def test_load_dataset_validation(self):
        """Test dataset validation after loading"""
        # Create test data with various data types
        test_data = pd.DataFrame({
            'week': pd.date_range('2023-01-01', periods=10, freq='W'),
            'İhtiyaç Kg': [100, 150, 200, None, 300, 250, 180, 220, 190, 160],
            'Stok Kodu': ['TEST001'] * 10,
            'invalid_col': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        })
        
        result = map_columns(test_data)
        
        # Check data types after mapping
        assert 'week' in result.columns
        assert 'quantity' in result.columns
        
        # Should handle NaN values appropriately
        if 'quantity' in result.columns:
            # Either NaN is kept or handled gracefully
            assert isinstance(result['quantity'].dtype, (type(np.dtype('float64')), type(np.dtype('int64'))))
