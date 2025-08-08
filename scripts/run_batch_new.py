#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modular Batch sales forecasting pipeline - Sheet-wise analysis version.

Usage (optional argument):
    python scripts/run_batch_new.py path/to/datafile.xlsx

If no argument is given:
    uploads/last_uploaded.json is read and the most recently uploaded dataset is used.

Key features:
- Each Excel sheet is analyzed separately 
- < 100 rows = Simple forecasting
- >= 100 rows = ML forecasting (RF, ET, GB, XGBoost, CatBoost)
- Modular design for different datasets
- Comprehensive PDF reporting with terminal outputs
- Product-level analysis and charts

Supported data formats:
    - .xlsx / .xls (Excel)
    - .csv
    - .sql        (SQLite script)
    - .sqlite / .db (SQLite database file)

Outputs:
    app/static/plots/   -> product-level PNG charts
    app/static/reports/ -> forecast_report.pdf, sheets_summary.csv, model forecasts
    project_root/       -> future_forecast.csv (RF predictions; legacy code compatibility)
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import pandas as pd
except ImportError:
    print("❌ Pandas is not installed. Please install with: pip install pandas")
    sys.exit(1)

# pyright: reportMissingModuleSource=false

# Import modules
from modules.config import (
    PROJECT_ROOT, UPLOADS_DIR, REPORTS_DIR, PLOTS_DIR,
    SHEET_ML_THRESHOLD, FORECAST_HORIZON_WEEKS, LIBRARY_AVAILABILITY
)
from modules.utils.metrics import (
    initialize_terminal_capture, log_output
)
from modules.data_loader import (
    load_dataset_generic, map_columns, prepare_weekly_data
)
from modules.feature_engineering import (
    create_features
)
from modules.forecasting.models import (
    simple_forecast_for_sheet, ml_forecast_for_sheet
)
from modules.improved_forecasting.models import (
    improved_forecast_for_sheet
)
from modules.visualization.charts import (
    create_product_analysis_charts, create_forecast_charts
)
from modules.visualization.pdf_reports import (
    create_comprehensive_pdf_report
)

def main():
    """
    Main execution function with sheet-wise analysis.
    """
    # Initialize terminal capture
    initialize_terminal_capture()
    
    log_output("🚀 Starting Modular Sheet-Based Sales Forecasting System...")
    log_output(f"📁 Project Directory: {PROJECT_ROOT}")
    log_output(f"🎯 ML Threshold: {SHEET_ML_THRESHOLD} rows")
    log_output(f"📅 Forecast Horizon: {FORECAST_HORIZON_WEEKS} weeks")
    
    # Library availability
    log_output(f"📚 Library Status:")
    log_output(f"   - XGBoost: {'✅' if LIBRARY_AVAILABILITY['xgboost'] else '❌'}")
    log_output(f"   - CatBoost: {'✅' if LIBRARY_AVAILABILITY['catboost'] else '❌'}")
    log_output(f"   - Prophet: {'✅' if LIBRARY_AVAILABILITY['prophet'] else '❌'}")
    log_output(f"   - Seaborn: {'✅' if LIBRARY_AVAILABILITY['seaborn'] else '❌'}")
    
    # Determine input file
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
        log_output(f"📄 File from command line: {input_file}")
    else:
        # Read from last_uploaded.json
        last_uploaded_path = UPLOADS_DIR / "last_uploaded.json"
        if last_uploaded_path.exists():
            with open(last_uploaded_path, 'r') as f:
                last_uploaded = json.load(f)
            # Check both possible key names for backward compatibility
            input_file = Path(last_uploaded.get("file_path") or last_uploaded.get("last_file"))
            log_output(f"📄 Last uploaded file: {input_file}")
        else:
            log_output("❌ No file specified and last_uploaded.json not found!")
            return
    
    if not input_file.exists():
        log_output(f"❌ File not found: {input_file}")
        return
    
    try:
        # Load datasets by sheet
        log_output("📂 Loading data...")
        sheets_data = load_dataset_generic(input_file)
        
        if not sheets_data:
            log_output("❌ No sheet data could be loaded!")
            return
        
        # Process each sheet
        all_forecasts = {}
        sheets_summary = []
        
        for sheet_name, sheet_df in sheets_data.items():
            log_output(f"\n{'='*60}")
            log_output(f"📄 Processing: {sheet_name} ({len(sheet_df)} rows)")
            log_output(f"{'='*60}")
            
            start_time = datetime.now()
            
            try:
                # Map columns
                mapped_df = map_columns(sheet_df)
                
                # Create product analysis charts
                create_product_analysis_charts(mapped_df, sheet_name)
                
                # Determine method based on data size
                if len(mapped_df) < SHEET_ML_THRESHOLD:
                    log_output(f"📊 {sheet_name}: Small data size ({len(mapped_df)} < {SHEET_ML_THRESHOLD}), using simple forecasting")
                    
                    # Simple forecasting
                    forecast_result = simple_forecast_for_sheet(mapped_df, sheet_name)
                    sheet_forecasts = {"simple": forecast_result}
                    method_used = "Simple"
                    
                else:
                    log_output(f"🤖 {sheet_name}: Sufficient data size ({len(mapped_df)} >= {SHEET_ML_THRESHOLD}), using ML models + improved method")
                    
                    # Complete ML pipeline: prepare weekly data -> create features -> ML forecast
                    weekly_data = prepare_weekly_data(mapped_df, sheet_name)
                    featured_data = create_features(weekly_data)
                    sheet_forecasts = ml_forecast_for_sheet(featured_data, sheet_name)
                    
                    # Add improved forecasting method
                    try:
                        log_output(f"🔧 {sheet_name}: Running improved statistical forecasting...")
                        improved_result = improved_forecast_for_sheet(mapped_df, sheet_name)
                        
                        # Convert to DataFrame format for consistency
                        improved_df = pd.DataFrame({
                            'sheet_name': [sheet_name] * FORECAST_HORIZON_WEEKS,
                            'week': list(range(1, FORECAST_HORIZON_WEEKS + 1)),
                            'forecast': improved_result
                        })
                        sheet_forecasts["improved"] = improved_df
                        log_output(f"✅ {sheet_name}: Improved forecasting completed. Average: {sum(improved_result)/len(improved_result):.2f}")
                        
                    except Exception as e:
                        log_output(f"⚠️ {sheet_name}: Improved forecasting failed: {e}")
                    
                    method_used = "ML+Improved"
                
                # Create forecast charts
                create_forecast_charts(sheet_forecasts, sheet_name, mapped_df)
                
                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Calculate average forecast and best model info
                forecast_mean = 0
                best_model_info = ""
                if sheet_forecasts:
                    first_forecast = list(sheet_forecasts.values())[0]
                    if len(first_forecast) > 0:
                        forecast_mean = first_forecast["forecast"].mean()
                        
                        # Get best model based on backtest if available
                        if hasattr(first_forecast, 'attrs'):
                            model_performances = []
                            for model_name, forecast_df in sheet_forecasts.items():
                                if hasattr(forecast_df, 'attrs') and forecast_df.attrs:
                                    mae = forecast_df.attrs.get('MAE', float('inf'))
                                    model_performances.append((model_name, mae))
                            
                            if model_performances:
                                best_model = min(model_performances, key=lambda x: x[1])
                                best_model_info = f"{best_model[0].upper()}(MAE:{best_model[1]:.2f})"
                
                # Store results
                all_forecasts[sheet_name] = sheet_forecasts
                
                sheets_summary.append({
                    "sheet_name": sheet_name,
                    "row_count": len(mapped_df),
                    "method": method_used,
                    "forecast_mean": forecast_mean,
                    "processing_time": processing_time,
                    "models_used": list(sheet_forecasts.keys()) if sheet_forecasts else [],
                    "best_model": best_model_info
                })
                
                log_output(f"✅ {sheet_name}: Processing completed ({processing_time:.2f}s)")
                
            except Exception as e:
                log_output(f"❌ {sheet_name}: Processing error: {e}")
                sheets_summary.append({
                    "sheet_name": sheet_name,
                    "row_count": len(sheet_df),
                    "method": "Error",
                    "forecast_mean": 0,
                    "processing_time": 0,
                    "models_used": [],
                    "best_model": ""
                })
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(sheets_summary)
        summary_path = REPORTS_DIR / "sheets_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        log_output(f"📊 Sheet summary saved: {summary_path}")
        
        # Save individual model forecasts
        log_output("\n📁 Saving model forecasts...")
        
        # Combine all forecasts by model
        combined_forecasts = {"rf": [], "et": [], "gb": [], "simple": [], "improved": []}
        
        # Add XGBoost and CatBoost if available
        if LIBRARY_AVAILABILITY['xgboost']:
            combined_forecasts["xgb"] = []
        if LIBRARY_AVAILABILITY['catboost']:
            combined_forecasts["cat"] = []
        
        for sheet_name, sheet_forecasts in all_forecasts.items():
            for model_name, forecast_df in sheet_forecasts.items():
                if model_name in combined_forecasts:
                    combined_forecasts[model_name].append(forecast_df)
        
        # Save combined forecasts
        for model_name, forecast_list in combined_forecasts.items():
            if forecast_list:
                combined_df = pd.concat(forecast_list, ignore_index=True)
                model_path = REPORTS_DIR / f"forecast_{model_name}.csv"
                combined_df.to_csv(model_path, index=False)
                log_output(f"💾 {model_name.upper()} forecasts saved: {model_path}")
        
        # Create legacy compatibility file (RF forecasts)
        if combined_forecasts["rf"]:
            rf_combined = pd.concat(combined_forecasts["rf"], ignore_index=True)
            legacy_path = PROJECT_ROOT / "future_forecast.csv"
            rf_combined.to_csv(legacy_path, index=False)
            log_output(f"🔄 Legacy compatibility file saved: {legacy_path}")
        
        # Create comprehensive PDF report
        create_comprehensive_pdf_report(summary_df, all_forecasts)
        
        # Final summary
        log_output(f"\n{'='*60}")
        log_output("🎉 ALL OPERATIONS COMPLETED!")
        log_output(f"{'='*60}")
        log_output(f"📊 Total processed sheets: {len(sheets_data)}")
        log_output(f"🤖 Processed with ML: {len(summary_df[summary_df['method'] == 'ML'])}")
        log_output(f"🔮 Processed with simple forecasting: {len(summary_df[summary_df['method'] == 'Simple'])}")
        log_output(f"❌ Errors: {len(summary_df[summary_df['method'] == 'Error'])}")
        log_output(f"📁 Output directory: {REPORTS_DIR}")
        log_output(f"📈 Charts directory: {PLOTS_DIR}")
        
        # Module summary
        log_output(f"\n📦 Modular Structure:")
        log_output(f"   - config.py: Configuration and parameters")
        log_output(f"   - data_loader.py: Data loading and column mapping")
        log_output(f"   - feature_engineering.py: Feature engineering")
        log_output(f"   - forecasting/models.py: ML models and forecasting")
        log_output(f"   - visualization/charts.py: Chart generation")
        log_output(f"   - visualization/pdf_reports.py: PDF report generation")
        log_output(f"   - utils/metrics.py: Metrics and utility functions")
        
    except Exception as e:
        log_output(f"❌ General error: {e}")
        import traceback
        log_output(f"❌ Details: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
