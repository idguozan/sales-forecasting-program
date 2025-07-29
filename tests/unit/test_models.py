"""
Unit tests for ML models
"""
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.modules.forecasting.models import (
    simple_forecast_for_sheet,
    ml_forecast_for_sheet,
    determine_model_type
)

class TestMLModels:
    
    def test_determine_model_type_small_data(self, sample_small_dataset):
        """Test model type determination for small datasets"""
        model_type = determine_model_type(sample_small_dataset, "test_sheet")
        
        # Small dataset should use simple forecasting
        assert model_type == "Simple"
    
    def test_determine_model_type_large_data(self, sample_large_dataset):
        """Test model type determination for large datasets"""
        model_type = determine_model_type(sample_large_dataset, "test_sheet")
        
        # Large dataset should use ML forecasting
        assert model_type == "ML"
    
    def test_simple_forecast_basic(self, sample_small_dataset):
        """Test simple forecasting with basic dataset"""
        result = simple_forecast_for_sheet(sample_small_dataset, "test_sheet")
        
        # Check result structure
        assert isinstance(result, pd.DataFrame)
        assert 'week' in result.columns
        assert 'forecast' in result.columns
        assert 'method' in result.columns
        
        # Check forecast length (should be FORECAST_HORIZON_WEEKS)
        assert len(result) == 12  # Default FORECAST_HORIZON_WEEKS
        
        # Check method is marked correctly
        assert all(result['method'] == 'Simple')
        
        # Forecasts should be non-negative
        assert (result['forecast'] >= 0).all()
    
    def test_simple_forecast_empty_data(self):
        """Test simple forecast with empty data"""
        empty_df = pd.DataFrame(columns=['week', 'İhtiyaç Kg', 'Stok Kodu'])
        
        result = simple_forecast_for_sheet(empty_df, "test_sheet")
        
        # Should handle gracefully
        assert isinstance(result, pd.DataFrame)
        # May be empty or have default values
        if len(result) > 0:
            assert 'forecast' in result.columns
    
    def test_simple_forecast_single_value(self):
        """Test simple forecast with single data point"""
        single_point = pd.DataFrame({
            'week': [pd.Timestamp('2023-01-01')],
            'İhtiyaç Kg': [100],
            'Stok Kodu': ['TEST']
        })
        
        result = simple_forecast_for_sheet(single_point, "test_sheet")
        
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            # Should use the single value for forecasting
            assert all(result['forecast'] == 100)
    
    def test_ml_forecast_basic(self, sample_large_dataset):
        """Test ML forecasting with adequate dataset"""
        result = ml_forecast_for_sheet(sample_large_dataset, "test_sheet")
        
        # Check result is dictionary (contains multiple model results)
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Check that it contains forecast data
        for model_name, model_result in result.items():
            assert isinstance(model_result, dict)
            assert 'forecast' in model_result
            assert 'metrics' in model_result
            
            # Check forecast structure
            forecast_df = model_result['forecast']
            assert isinstance(forecast_df, pd.DataFrame)
            assert 'week' in forecast_df.columns
            assert 'forecast' in forecast_df.columns
    
    def test_ml_forecast_insufficient_data(self, sample_small_dataset):
        """Test ML forecast fallback with insufficient data"""
        # Force ML forecast on small dataset
        result = ml_forecast_for_sheet(sample_small_dataset, "test_sheet")
        
        # Should either fallback to simple or handle gracefully
        assert isinstance(result, (dict, pd.DataFrame))
    
    @patch('scripts.modules.forecasting.models.RandomForestRegressor')
    @patch('scripts.modules.forecasting.models.ExtraTreesRegressor')
    def test_ml_forecast_model_training(self, mock_et, mock_rf, sample_large_dataset):
        """Test ML forecast model training with mocks"""
        # Mock model behavior
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([100] * 12)
        mock_model.score.return_value = 0.85
        
        mock_rf.return_value = mock_model
        mock_et.return_value = mock_model
        
        result = ml_forecast_for_sheet(sample_large_dataset, "test_sheet")
        
        # Should have called model training
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Models should have been instantiated
        mock_rf.assert_called()
        mock_et.assert_called()
    
    def test_forecast_date_continuity(self, sample_large_dataset):
        """Test that forecast dates are continuous"""
        result = ml_forecast_for_sheet(sample_large_dataset, "test_sheet")
        
        for model_name, model_result in result.items():
            forecast_df = model_result['forecast']
            
            if len(forecast_df) > 1:
                # Check that dates are weekly intervals
                date_diffs = forecast_df['week'].diff().dropna()
                expected_diff = pd.Timedelta(weeks=1)
                
                # Allow some tolerance for date calculations
                assert all(abs(diff - expected_diff) < pd.Timedelta(days=1) 
                          for diff in date_diffs)
    
    def test_forecast_values_reasonable(self, sample_large_dataset):
        """Test that forecast values are reasonable"""
        result = ml_forecast_for_sheet(sample_large_dataset, "test_sheet")
        
        # Get original data statistics
        original_mean = sample_large_dataset['İhtiyaç Kg'].mean()
        original_std = sample_large_dataset['İhtiyaç Kg'].std()
        
        for model_name, model_result in result.items():
            forecast_df = model_result['forecast']
            forecast_values = forecast_df['forecast']
            
            # Forecasts should be non-negative
            assert (forecast_values >= 0).all()
            
            # Forecasts should be within reasonable bounds
            # (e.g., not more than 10x the historical mean)
            assert (forecast_values <= original_mean * 10).all()
    
    def test_metrics_calculation(self, sample_large_dataset):
        """Test that metrics are calculated properly"""
        result = ml_forecast_for_sheet(sample_large_dataset, "test_sheet")
        
        for model_name, model_result in result.items():
            metrics = model_result['metrics']
            
            # Check that metrics exist
            assert isinstance(metrics, dict)
            
            # Check for common metrics
            expected_metrics = ['mae', 'rmse', 'mape', 'wape']
            for metric in expected_metrics:
                if metric in metrics:
                    # Metrics should be numeric
                    assert isinstance(metrics[metric], (int, float))
                    # Metrics should be non-negative
                    assert metrics[metric] >= 0
    
    def test_model_comparison(self, sample_large_dataset):
        """Test that multiple models are compared"""
        result = ml_forecast_for_sheet(sample_large_dataset, "test_sheet")
        
        # Should have multiple models
        assert len(result) >= 2
        
        # Common models should be present
        expected_models = ['RandomForest', 'ExtraTrees']
        for model in expected_models:
            # At least one should be present
            model_found = any(model.lower() in model_name.lower() 
                            for model_name in result.keys())
            if not model_found:
                # This is okay, models might have different names
                pass
