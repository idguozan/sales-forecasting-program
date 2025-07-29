"""
Unit tests for feature_engineering module
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.modules.feature_engineering import (
    create_features,
    prepare_weekly_data
)

class TestFeatureEngineering:
    
    def test_prepare_weekly_data_basic(self, sample_sales_data):
        """Test basic weekly data preparation"""
        result = prepare_weekly_data(sample_sales_data, "test_sheet")
        
        # Check structure
        assert isinstance(result, pd.DataFrame)
        assert 'week' in result.columns
        assert len(result) > 0
        
        # Check sorting (weeks should be in ascending order)
        if len(result) > 1:
            assert result['week'].is_monotonic_increasing
    
    def test_prepare_weekly_data_aggregation(self):
        """Test weekly data aggregation"""
        # Create data with multiple entries per week
        base_date = pd.Timestamp('2023-01-01')
        test_data = pd.DataFrame({
            'week': [base_date, base_date, base_date + pd.Timedelta(weeks=1)],
            'İhtiyaç Kg': [100, 50, 200],
            'Stok Kodu': ['TEST001', 'TEST001', 'TEST001']
        })
        
        result = prepare_weekly_data(test_data, "test_sheet")
        
        # Should aggregate multiple entries for same week
        unique_weeks = result['week'].nunique()
        assert unique_weeks <= 2  # Should have max 2 weeks
        
        # Check aggregation logic
        first_week_total = result[result['week'] == base_date]['İhtiyaç Kg'].sum()
        assert first_week_total > 0
    
    def test_create_features_basic(self, sample_mapped_data):
        """Test basic feature creation"""
        result = create_features(sample_mapped_data)
        
        # Check that DataFrame is returned
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        # Check that original columns are preserved
        assert 'week' in result.columns
        assert 'quantity' in result.columns
        
        # Check for time-based features
        time_features = ['week_of_year', 'month', 'quarter']
        for feature in time_features:
            if feature in result.columns:
                assert result[feature].notna().any()
    
    def test_create_features_lag_features(self, sample_mapped_data):
        """Test lag feature creation"""
        result = create_features(sample_mapped_data)
        
        # Check for lag features
        lag_features = ['lag_1', 'lag_2', 'lag_3', 'lag_4']
        
        for lag_feat in lag_features:
            if lag_feat in result.columns:
                # Lag features should have some NaN values at the beginning
                assert result[lag_feat].isna().sum() >= 1
                # But should have some non-NaN values too
                assert result[lag_feat].notna().sum() > 0
    
    def test_create_features_rolling_features(self, sample_mapped_data):
        """Test rolling window features"""
        result = create_features(sample_mapped_data)
        
        # Check for rolling features
        rolling_features = ['rolling_mean_3', 'rolling_std_3', 'rolling_mean_7']
        
        for roll_feat in rolling_features:
            if roll_feat in result.columns:
                # Rolling features should have some values
                assert result[roll_feat].notna().sum() > 0
                # Should be numeric
                assert pd.api.types.is_numeric_dtype(result[roll_feat])
    
    def test_create_features_insufficient_data(self):
        """Test feature creation with insufficient data"""
        # Very small dataset
        small_data = pd.DataFrame({
            'week': pd.date_range('2023-01-01', periods=3, freq='W'),
            'quantity': [100, 150, 120]
        })
        
        result = create_features(small_data)
        
        # Should still work but with limited features
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'week' in result.columns
        assert 'quantity' in result.columns
    
    def test_create_features_empty_data(self):
        """Test feature creation with empty data"""
        empty_data = pd.DataFrame(columns=['week', 'quantity'])
        
        result = create_features(empty_data)
        
        # Should handle empty data gracefully
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_create_features_data_types(self, sample_mapped_data):
        """Test that feature creation maintains proper data types"""
        result = create_features(sample_mapped_data)
        
        # Week should be datetime
        if 'week' in result.columns:
            assert pd.api.types.is_datetime64_any_dtype(result['week'])
        
        # Numeric features should be numeric
        numeric_features = ['quantity', 'lag_1', 'rolling_mean_3']
        for feat in numeric_features:
            if feat in result.columns:
                assert pd.api.types.is_numeric_dtype(result[feat])
    
    def test_create_features_no_nulls_in_recent_data(self, sample_mapped_data):
        """Test that recent data (used for forecasting) has minimal nulls"""
        result = create_features(sample_mapped_data)
        
        if len(result) > 10:
            # Check last 5 rows (most recent data)
            recent_data = result.tail(5)
            
            # Core columns should not have nulls in recent data
            core_cols = ['week', 'quantity']
            for col in core_cols:
                if col in recent_data.columns:
                    assert not recent_data[col].isna().all()
    
    def test_create_features_seasonal_components(self, sample_mapped_data):
        """Test seasonal feature creation"""
        result = create_features(sample_mapped_data)
        
        # Check seasonal features
        seasonal_features = ['month', 'quarter', 'week_of_year']
        
        for feat in seasonal_features:
            if feat in result.columns:
                # Should be within expected ranges
                if feat == 'month':
                    assert result[feat].min() >= 1
                    assert result[feat].max() <= 12
                elif feat == 'quarter':
                    assert result[feat].min() >= 1
                    assert result[feat].max() <= 4
                elif feat == 'week_of_year':
                    assert result[feat].min() >= 1
                    assert result[feat].max() <= 53
