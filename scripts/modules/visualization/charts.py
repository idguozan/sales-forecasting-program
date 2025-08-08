#!/usr/bin/env python3
 # -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization module for the sales forecasting system.
Contains functions for creating charts and plots.
"""

import matplotlib  # pyright: ignore[reportMissingModuleSource]
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]
import pandas as pd  # pyright: ignore[reportMissingModuleSource]
import numpy as np  # pyright: ignore[reportMissingImports]
from typing import Dict

from ..config import PLOTS_DIR, LIBRARY_AVAILABILITY
from ..utils.metrics import log_output

# Set seaborn style if available
if LIBRARY_AVAILABILITY['seaborn']:
    import seaborn as sns  # type: ignore
    sns.set_style("whitegrid")

def create_product_analysis_charts(df: pd.DataFrame, sheet_name: str):
    """
    Create product-level analysis charts.
    """
    log_output(f"📊 {sheet_name}: Creating product analysis charts...")
    
    try:
        # Create total sales column
        df["TotalSales"] = df["Quantity"] * df["UnitPrice"]
        
        # Product sales analysis
        product_sales = df.groupby("StockCode").agg({
            "TotalSales": "sum",
            "Quantity": "sum"
        }).reset_index()
        
        product_sales = product_sales.sort_values("TotalSales", ascending=False).head(20)
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Product Analysis - {sheet_name}', fontsize=16, fontweight='bold')
        
        # Top 10 products by sales
        top_products = product_sales.head(10)
        ax1.barh(range(len(top_products)), top_products["TotalSales"])
        ax1.set_yticks(range(len(top_products)))
        ax1.set_yticklabels(top_products["StockCode"], fontsize=8)
        ax1.set_xlabel("Total Sales")
        ax1.set_title("Top 10 Best Selling Products")
        ax1.grid(True, alpha=0.3)
        
        # Monthly sales trend
        df["Month"] = df["InvoiceDate"].dt.to_period("M")
        monthly_sales = df.groupby("Month")["TotalSales"].sum()
        ax2.plot(range(len(monthly_sales)), monthly_sales.values, marker='o')
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Total Sales")
        ax2.set_title("Monthly Sales Trend")
        ax2.grid(True, alpha=0.3)
        
        # Sales distribution
        ax3.hist(df["TotalSales"], bins=50, alpha=0.7, edgecolor='black')
        ax3.set_xlabel("Sales Amount")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Sales Distribution")
        ax3.grid(True, alpha=0.3)
        
        # Product count by price range
        price_ranges = pd.cut(df["UnitPrice"], bins=5)
        price_counts = price_ranges.value_counts()
        ax4.pie(price_counts.values, labels=[f"{interval}" for interval in price_counts.index], autopct='%1.1f%%')
        ax4.set_title("Product Distribution by Price Range")
        
        plt.tight_layout()
        
        # Save chart
        chart_path = PLOTS_DIR / f"product_analysis_{sheet_name.replace(' ', '_')}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        log_output(f"✅ {sheet_name}: Product analysis chart saved: {chart_path}")
        
    except Exception as e:
        log_output(f"❌ {sheet_name}: Could not create product analysis chart: {e}")

def create_forecast_charts(forecasts: Dict[str, pd.DataFrame], sheet_name: str, historical_data: pd.DataFrame):
    """
    Create forecast visualization charts with separate historical and future visuals.
    """
    log_output(f"📈 {sheet_name}: Creating forecast charts...")
    
    try:
        # Prepare historical data
        historical_data["TotalSales"] = historical_data["Quantity"] * historical_data["UnitPrice"]
        historical_data["week"] = historical_data["InvoiceDate"].dt.to_period("W").dt.start_time
        weekly_historical = historical_data.groupby("week")["TotalSales"].sum().reset_index()
        
        # Create figure with 2x2 layout 
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Sales Analysis and Forecasts - {sheet_name}', fontsize=16, fontweight='bold')
        
        # Historical data visualization (Top left)
        ax1 = axes[0, 0]
        ax1.plot(weekly_historical["week"], weekly_historical["TotalSales"], 
                color='darkblue', marker='o', markersize=3, linewidth=2, label="Historical Sales")
        ax1.fill_between(weekly_historical["week"], weekly_historical["TotalSales"], 
                        alpha=0.3, color='lightblue')
        ax1.set_xlabel("Week")
        ax1.set_ylabel("Sales")
        ax1.set_title("📊 Historical Sales Performance", fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Future forecasts visualization (Top right)
        ax2 = axes[0, 1]
        colors = {"rf": "#FF6B6B", "et": "#4ECDC4", "gb": "#45B7D1", "simple": "#FFA07A", "improved": "#FFD700", "xgb": "#1E90FF", "cat": "#8B008B"}

        # Forecast horizon belirleme (genelde 12 hafta)
        forecast_horizon = 12
        
        # Gelecek haftaları sıralı bir şekilde oluştur
        future_week_numbers = list(range(1, forecast_horizon + 1))
        
        # Tüm modellerin tahminlerini aynı X değerleri ile çiz
        for model_name, forecast_df in forecasts.items():
            if len(forecast_df) > 0 and "forecast" in forecast_df.columns:
                color = colors.get(model_name, "#9B59B6")
                
                # Sadece forecast değerlerini al, X için sıralı numaralar kullan
                forecast_values = forecast_df["forecast"].values[:forecast_horizon]
                x_values = future_week_numbers[:len(forecast_values)]
                
                ax2.plot(x_values, forecast_values, 
                        label=f"{model_name.upper()} Forecast", 
                        marker='s', markersize=4, color=color, linewidth=2)
                # Add fill for better visibility
                ax2.fill_between(x_values, forecast_values, 
                               alpha=0.2, color=color)

        ax2.set_xlabel("Future Week")
        ax2.set_ylabel("Predicted Sales")
        ax2.set_title("🔮 Future Sales Forecasts", fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0.5, forecast_horizon + 0.5])  # X eksenini 1-12 hafta arası ayarla
        
        # Model comparison (Bottom left)
        ax3 = axes[1, 0]
        model_means = {}
        for model_name, forecast_df in forecasts.items():
            if len(forecast_df) > 0:
                model_means[model_name.upper()] = forecast_df["forecast"].mean()
        
        if model_means:
            models = list(model_means.keys())
            means = list(model_means.values())
            bars = ax3.bar(models, means, color=[colors.get(m.lower(), "#9B59B6") for m in models])
            ax3.set_ylabel("Average Forecast")
            ax3.set_title("🏆 Model Comparison", fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mean:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Statistical summary (Bottom right)
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary text
        summary_text = f"📋 Data Summary - {sheet_name}\n\n"
        summary_text += f"• Total Sales Records: {len(historical_data):,}\n"
        summary_text += f"• Weekly Data Points: {len(weekly_historical)}\n"
        summary_text += f"• Average Weekly Sales: {weekly_historical['TotalSales'].mean():,.0f}\n"
        summary_text += f"• Highest Weekly Sales: {weekly_historical['TotalSales'].max():,.0f}\n"
        summary_text += f"• Lowest Weekly Sales: {weekly_historical['TotalSales'].min():,.0f}\n"
        summary_text += f"• Forecast Horizon: 12 weeks\n\n"
        
        if model_means:
            best_model = max(model_means.keys(), key=lambda x: model_means[x])
            summary_text += f"🏆 Highest Forecast: {best_model}\n"
            summary_text += f"   Average: {model_means[best_model]:,.0f}"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save chart
        chart_path = PLOTS_DIR / f"forecast_{sheet_name.replace(' ', '_')}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        log_output(f"✅ {sheet_name}: Forecast chart saved: {chart_path}")
        
    except Exception as e:
        log_output(f"❌ {sheet_name}: Could not create forecast chart: {e}")

def create_backtest_chart(forecast_df: pd.DataFrame, historical_data: pd.DataFrame, 
                         model_name: str, sheet_name: str, metrics: Dict) -> plt.Figure:
    """
    Create a backtest visualization chart for a specific model.
    """
    try:
        # Prepare historical weekly data
        historical_data["TotalSales"] = historical_data["Quantity"] * historical_data["UnitPrice"]
        historical_data["week"] = historical_data["InvoiceDate"].dt.to_period("W").dt.start_time
        weekly_historical = historical_data.groupby("week")["TotalSales"].sum().reset_index()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f'{model_name.upper()} Backtest - {sheet_name}', fontsize=14, fontweight='bold')
        
        # Historical vs Forecast comparison
        ax1.plot(weekly_historical["week"], weekly_historical["TotalSales"], 
                label="Gerçek Satışlar", marker='o', markersize=3, linewidth=2, color='darkblue')
        
        if len(forecast_df) > 0:
            ax1.plot(forecast_df["week"], forecast_df["forecast"], 
                    label=f"{model_name.upper()} Tahmini", marker='s', markersize=4, 
                    linewidth=2, color='red', linestyle='--')
        
        ax1.set_xlabel("Hafta")
        ax1.set_ylabel("Satış")
        ax1.set_title("Gerçek vs Tahmin Karşılaştırması")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Error distribution
        if len(forecast_df) > 0 and len(weekly_historical) > 0:
            # Calculate errors for overlapping weeks
            merged = pd.merge(weekly_historical, forecast_df.rename(columns={'forecast': 'predicted'}), 
                            on='week', how='inner')
            if len(merged) > 0:
                errors = merged['TotalSales'] - merged['predicted']
                ax2.hist(errors, bins=20, alpha=0.7, edgecolor='black', color='lightcoral')
                ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
                ax2.set_xlabel("Hata (Gerçek - Tahmin)")
                ax2.set_ylabel("Frekans")
                ax2.set_title("Hata Dağılımı")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # Add metrics text
        if metrics:
            metrics_text = f"MAE: {metrics.get('MAE', 0):.2f}\n"
            metrics_text += f"RMSE: {metrics.get('RMSE', 0):.2f}\n"
            metrics_text += f"MAPE: {metrics.get('MAPE', 0):.1f}%\n"
            metrics_text += f"WAPE: {metrics.get('WAPE', 0):.1f}%"
            
            ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        log_output(f"❌ Backtest grafiği oluşturulamadı: {e}")
        # Return empty figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, f"Grafik oluşturulamadı\n{str(e)}", 
               ha='center', va='center', transform=ax.transAxes)
        return fig
