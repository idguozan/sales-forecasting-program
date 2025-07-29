"""
Integration tests for the complete forecasting pipeline
"""
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.modules.data_loader import load_dataset_generic, map_columns, prepare_weekly_data
from scripts.modules.feature_engineering import create_features
from scripts.modules.forecasting.models import simple_forecast_for_sheet, ml_forecast_for_sheet

class TestPipelineIntegration:
    
    def test_complete_pipeline_small_dataset(self, sample_excel_file):
        """Test complete pipeline with small dataset (simple forecasting)"""
        # Step 1: Load data
        datasets = load_dataset_generic(Path(sample_excel_file))
        assert len(datasets) > 0
        
        # Step 2: Process each sheet
        for sheet_name, raw_data in datasets.items():
            # Map columns
            mapped_data = map_columns(raw_data)
            assert len(mapped_data) > 0
            
            # Since test data is small, should use simple forecasting
            forecast_result = simple_forecast_for_sheet(mapped_data, sheet_name)
            
            # Validate final result
            assert isinstance(forecast_result, pd.DataFrame)
            assert 'forecast' in forecast_result.columns
            assert len(forecast_result) > 0
    
    def test_complete_pipeline_large_dataset(self, sample_large_dataset):
        """Test complete pipeline with large dataset (ML forecasting)"""
        # Create temporary Excel file with large dataset
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            sample_large_dataset.to_excel(tmp.name, sheet_name='Large_Sheet', index=False)
            tmp_path = tmp.name
        
        try:
            # Step 1: Load data
            datasets = load_dataset_generic(Path(tmp_path))
            assert len(datasets) > 0
            
            # Step 2: Process each sheet
            for sheet_name, raw_data in datasets.items():
                # Map columns
                mapped_data = map_columns(raw_data)
                assert len(mapped_data) > 0
                
                # Prepare weekly data (missing step!)
                weekly_data = prepare_weekly_data(mapped_data, sheet_name)
                assert len(weekly_data) > 0
                assert 'week' in weekly_data.columns
                
                # Feature engineering
                featured_data = create_features(weekly_data)
                assert len(featured_data) > 0
                
                # ML forecasting (should trigger for large dataset)
                forecast_result = ml_forecast_for_sheet(featured_data, sheet_name)
                
                # Validate final result
                assert isinstance(forecast_result, dict)
                assert len(forecast_result) > 0
                
                # Check each model result (DataFrame formatında)
                for model_name, model_result in forecast_result.items():
                    assert isinstance(model_result, pd.DataFrame)
                    assert len(model_result) > 0
                    assert 'week' in model_result.columns
                    assert 'forecast' in model_result.columns
        
        finally:
            os.unlink(tmp_path)
    
    def test_pipeline_data_consistency(self, sample_sales_data):
        """Test data consistency through pipeline stages"""
        original_length = len(sample_sales_data)
        
        # Stage 1: Column mapping
        mapped_data = map_columns(sample_sales_data)
        
        # Data should not be lost in mapping
        assert len(mapped_data) <= original_length  # May filter invalid rows
        assert len(mapped_data) > 0
        
        # Stage 2: Weekly data preparation
        weekly_data = prepare_weekly_data(mapped_data, "test_sheet")
        assert len(weekly_data) > 0
        assert 'week' in weekly_data.columns
        
        # Stage 3: Feature engineering
        featured_data = create_features(weekly_data)
        
        # Features may create some NaN rows that get dropped
        assert len(featured_data) <= len(weekly_data)
        
        # But should retain most data
        retention_rate = len(featured_data) / len(weekly_data)
        assert retention_rate > 0.5  # At least 50% data retention
    
    def test_pipeline_error_handling(self):
        """Test pipeline behavior with problematic data"""
        # Test with various problematic datasets
        
        # Empty dataset
        empty_df = pd.DataFrame()
        try:
            mapped_empty = map_columns(empty_df)
            assert isinstance(mapped_empty, pd.DataFrame)
        except Exception as e:
            # Should handle gracefully
            assert "empty" in str(e).lower() or "no data" in str(e).lower()
        
        # Dataset with all NaN values
        nan_df = pd.DataFrame({
            'week': [None, None, None],
            'İhtiyaç Kg': [None, None, None],
            'Stok Kodu': [None, None, None]
        })
        
        try:
            mapped_nan = map_columns(nan_df)
            # Should either handle or fail gracefully
            if isinstance(mapped_nan, pd.DataFrame):
                assert len(mapped_nan) <= 3
        except Exception:
            # Acceptable to fail with all-NaN data
            pass
    
    def test_pipeline_performance(self, sample_large_dataset):
        """Test pipeline performance with larger dataset"""
        import time
        
        start_time = time.time()
        
        # Run complete pipeline
        mapped_data = map_columns(sample_large_dataset)
        weekly_data = prepare_weekly_data(mapped_data, "perf_test")
        featured_data = create_features(weekly_data)
        
        # Don't run full ML (too slow for tests), just measure preprocessing
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert processing_time < 10.0  # 10 seconds max for preprocessing
        
        # Data should be processed efficiently
        rows_per_second = len(sample_large_dataset) / processing_time
        assert rows_per_second > 10  # At least 10 rows per second
    
    def test_pipeline_output_format(self, sample_sales_data):
        """Test that pipeline outputs are in expected format"""
        # Process through pipeline
        mapped_data = map_columns(sample_sales_data)
        
        # Simple forecast
        simple_result = simple_forecast_for_sheet(mapped_data, "test_sheet")
        
        # Check output format
        assert isinstance(simple_result, pd.DataFrame)
        required_cols = ['week', 'forecast', 'method']
        for col in required_cols:
            assert col in simple_result.columns
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(simple_result['week'])
        assert pd.api.types.is_numeric_dtype(simple_result['forecast'])
        assert simple_result['method'].dtype == 'object'  # String
    
    def test_pipeline_multiple_sheets(self):
        """Test pipeline with multiple sheets/products"""
        # Create test data with multiple products
        base_dates = pd.date_range('2023-01-01', periods=30, freq='W')
        
        datasets = {}
        for i, product in enumerate(['PRODUCT_A', 'PRODUCT_B', 'PRODUCT_C']):
            datasets[product] = pd.DataFrame({
                'week': base_dates,
                'İhtiyaç Kg': np.random.randint(50, 200, 30) + i * 50,
                'Stok Kodu': [product] * 30
            })
        
        # Process each dataset
        results = {}
        for sheet_name, raw_data in datasets.items():
            mapped_data = map_columns(raw_data)
            forecast_result = simple_forecast_for_sheet(mapped_data, sheet_name)
            results[sheet_name] = forecast_result
        
        # Should have results for all products
        assert len(results) == 3
        
        # Each result should be valid
        for product, result in results.items():
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert 'forecast' in result.columns
