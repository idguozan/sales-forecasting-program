#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Improved Statistical Forecasting Models
"""


# pyright: reportMissingModuleSource=false
# pyright: reportMissingImports=false
try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"❌ Required library not found: {e}")
    print("Please install with: pip install pandas numpy")
    import sys
    sys.exit(1)

import warnings
warnings.filterwarnings('ignore')

def analyze_product_pattern(df, sheet_name):
    """Ürün pattern analizi"""
    
    # Ensure date column is datetime
    if 'InvoiceDate' in df.columns:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df = df.sort_values('InvoiceDate')
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    
    # Use Quantity column (standard pipeline format) or total_quantity
    quantity_col = None
    if 'Quantity' in df.columns:
        quantity_col = 'Quantity'
    elif 'total_quantity' in df.columns:
        quantity_col = 'total_quantity'
    
    if not quantity_col:
        return None
    
    values = df[quantity_col].values
    
    # Basic statistics
    mean_val = np.mean(values)
    median_val = np.median(values)
    std_val = np.std(values)
    cv = std_val / mean_val if mean_val > 0 else 999
    
    # Robust statistics
    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25
    
    # Zero/low analysis
    zero_ratio = np.sum(values == 0) / len(values)
    low_threshold = mean_val * 0.1 if mean_val > 0 else 0
    low_ratio = np.sum(values < low_threshold) / len(values)
    
    # Recent trend analysis
    if len(values) >= 8:
        recent_8 = values[-8:]
        older_8 = values[-16:-8] if len(values) >= 16 else values[:-8]
        trend_factor = np.mean(recent_8) / np.mean(older_8) if np.mean(older_8) > 0 else 1
    else:
        trend_factor = 1
    
    # Volatility category
    if zero_ratio > 0.3:
        category = "SPARSE"
    elif cv < 0.4:
        category = "STABLE"
    elif cv < 0.8:
        category = "MODERATE"
    else:
        category = "VOLATILE"
    
    # Recent pattern
    last_4 = values[-4:] if len(values) >= 4 else values
    
    stats = {
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'cv': cv,
        'q25': q25,
        'q75': q75,
        'iqr': iqr,
        'trend_factor': trend_factor,
        'category': category,
        'zero_ratio': zero_ratio,
        'low_ratio': low_ratio,
        'last_4': last_4,
        'recent_avg': np.mean(last_4),
        'all_values': values
    }
    
    return stats

def smart_forecast(stats, periods=12):
    """Akıllı tahmin - pipeline format için"""
    
    if not stats:
        return [0] * periods
    
    predictions = []
    
    # Base prediction hesapla
    if stats['category'] == "SPARSE":
        # Sparse: very conservative
        if stats['zero_ratio'] > 0.5:
            base_pred = stats['median'] * 0.5
        else:
            base_pred = stats['recent_avg'] * 0.7
            
    elif stats['category'] == "STABLE":
        # Stable: recent average with small trend
        base_pred = stats['recent_avg']
        if 0.8 <= stats['trend_factor'] <= 1.2:  # Reasonable trend
            base_pred *= stats['trend_factor']
            
    elif stats['category'] == "MODERATE":
        # Moderate: blend recent with robust mean
        base_pred = 0.7 * stats['recent_avg'] + 0.3 * stats['median']
        if 0.7 <= stats['trend_factor'] <= 1.5:  # Apply trend if reasonable
            base_pred *= stats['trend_factor']
            
    else:  # VOLATILE
        # Volatile: median-based with recent signal
        base_pred = stats['median']
        
        # Check if recent trend is strong and consistent
        if len(stats['last_4']) >= 3:
            recent_trend = stats['last_4'][-1] / stats['last_4'][-2] if stats['last_4'][-2] > 0 else 1
            if 0.5 <= recent_trend <= 2.0:  # Reasonable recent change
                base_pred = 0.6 * stats['median'] + 0.4 * stats['recent_avg']
    
    # Universal safety constraints
    min_limit = max(0, stats['q25'] * 0.3)
    max_limit = stats['q75'] * 2.5
    
    # Additional constraint for high CV products
    if stats['cv'] > 1.5:
        max_limit = min(max_limit, stats['median'] * 3.0)
    
    # Her hafta için farklı tahmin üret
    for period in range(periods):
        
        # Haftalık varyasyon ekle
        week_factor = 1.0
        
        # Mevsimsel/haftalık pattern simülasyonu
        if stats['category'] == "STABLE":
            # Stable products: minimal variation
            variation = 0.95 + (period % 4) * 0.025  # 0.95-1.025 arası
            week_factor = variation
            
        elif stats['category'] == "MODERATE":
            # Moderate products: some weekly variation
            weekly_pattern = [1.0, 0.95, 1.05, 0.98, 1.02, 0.97, 1.03, 1.01, 0.99, 1.04, 0.96, 1.02]
            week_factor = weekly_pattern[period]
            
        elif stats['category'] == "VOLATILE":
            # Volatile products: more variation but controlled
            import random
            random.seed(42 + period)  # Deterministic but varied
            week_factor = 0.85 + random.random() * 0.3  # 0.85-1.15 arası
            
        else:  # SPARSE
            # Sparse products: occasional spikes
            if period in [2, 5, 8, 11]:  # Some weeks have higher activity
                week_factor = 1.2
            else:
                week_factor = 0.9
        
        # Trend decay over time
        if stats['trend_factor'] != 1.0:
            trend_decay = max(0.8, 1.0 - (period * 0.02))  # Trend etkisi zamanla azalır
            trend_adjusted_factor = 1.0 + (stats['trend_factor'] - 1.0) * trend_decay
            week_factor *= trend_adjusted_factor
        
        # Final prediction for this week
        weekly_pred = base_pred * week_factor
        
        # Apply constraints
        final_pred = np.clip(weekly_pred, min_limit, max_limit)
        final_pred = max(0, float(final_pred))  # Convert to Python float
        
        predictions.append(final_pred)
    
    return predictions

def calculate_confidence(stats):
    """Tahmin güven skoru hesapla"""
    
    if not stats:
        return 50
    
    confidence = 100  # Start with 100%
    
    # CV penalty
    if stats['cv'] > 2.0:
        confidence -= 40
    elif stats['cv'] > 1.5:
        confidence -= 30
    elif stats['cv'] > 1.0:
        confidence -= 20
    elif stats['cv'] > 0.5:
        confidence -= 10
    
    # Zero ratio penalty
    if stats['zero_ratio'] > 0.3:
        confidence -= 30
    elif stats['zero_ratio'] > 0.1:
        confidence -= 15
    
    # Data amount bonus/penalty
    data_points = len(stats['all_values'])
    if data_points < 20:
        confidence -= 20
    elif data_points < 50:
        confidence -= 10
    elif data_points > 100:
        confidence += 10
    
    return max(20, min(100, confidence))  # Between 20-100%

def improved_forecast_for_sheet(df, sheet_name):
    """Ana pipeline için improved forecasting fonksiyonu"""
    
    try:
        # Pattern analizi
        stats = analyze_product_pattern(df, sheet_name)
        
        if not stats:
            # Fallback: simple average
            quantity_col = None
            if 'Quantity' in df.columns:
                quantity_col = 'Quantity'
            elif 'total_quantity' in df.columns:
                quantity_col = 'total_quantity'
            
            if quantity_col:
                avg_demand = df[quantity_col].mean()
                return [max(0, avg_demand)] * 12
            else:
                return [0] * 12
        
        # Smart forecast
        predictions = smart_forecast(stats, periods=12)
        
        # Confidence calculation
        confidence = calculate_confidence(stats)
        
        # Debug output
        print(f"   📈 Improved Stats - Category: {stats['category']}, CV: {stats['cv']:.2f}, Confidence: {confidence:.1f}%")
        print(f"   📊 Predictions: {[round(p, 2) for p in predictions[:3]]}... (first 3)")
        
        # Return predictions as simple list for compatibility
        return predictions
        
    except Exception as e:
        print(f"   ❌ Improved forecasting error for {sheet_name}: {e}")
        # Fallback
        quantity_col = None
        if 'Quantity' in df.columns:
            quantity_col = 'Quantity'
        elif 'total_quantity' in df.columns:
            quantity_col = 'total_quantity'
        
        if quantity_col:
            avg_demand = df[quantity_col].mean()
            return [max(0, avg_demand)] * 12
        else:
            return [0] * 12
