#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced PDF reporting module for the sales forecasting system.
Includes baseline vs optimized comparison and detailed analytics.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import io
import contextlib
from pathlib import Path

from ..config import REPORTS_DIR, FORECAST_HORIZON_WEEKS, LIBRARY_AVAILABILITY
from ..utils.metrics import log_output, get_terminal_contents
from ..visualization.charts import create_backtest_chart

def create_comprehensive_pdf_report(sheets_summary: pd.DataFrame, all_forecasts: Dict):
    """
    Create comprehensive PDF report with terminal outputs and all charts.
    Now uses current conservative optimization results from live data.
    """
    log_output("📄 Creating comprehensive PDF report...")
    
    try:
        pdf_path = REPORTS_DIR / "comprehensive_forecast_report.pdf"
        
        with PdfPages(pdf_path) as pdf:
            # Page 1: Summary Report
            _create_summary_page(pdf, sheets_summary)
            
            # Page 2: Terminal Outputs
            _create_terminal_outputs_page(pdf)
            
            # Page 3: Detailed Sheet Summary Table
            _create_detailed_summary_table(pdf, sheets_summary)
            
            # Pages 4+: Individual Sheet Backtest Results
            _create_backtest_pages(pdf, all_forecasts, sheets_summary)
            
        log_output(f"✅ PDF report created: {pdf_path}")
        
    except Exception as e:
        log_output(f"❌ Could not create PDF report: {e}")

def _create_summary_page(pdf: PdfPages, sheets_summary: pd.DataFrame):
    """Create the summary page of the PDF report"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sales Forecasting System - Summary Report', fontsize=16, fontweight='bold')
    
    # Sheet analysis summary
    method_counts = sheets_summary['method'].value_counts()
    ax1.pie(method_counts.values, labels=method_counts.index, autopct='%1.1f%%')
    ax1.set_title('Method Distribution')
    
    # Row count distribution
    ax2.hist(sheets_summary['row_count'], bins=20, alpha=0.7, edgecolor='black')
    ax2.axvline(x=100, color='red', linestyle='--', label='ML Threshold (100)')
    ax2.set_xlabel('Row Count')
    ax2.set_ylabel('Sheet Count')
    ax2.set_title('Row Count Distribution per Sheet')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Processing time by method (if available)
    if 'processing_time' in sheets_summary.columns:
        method_time = sheets_summary.groupby('method')['processing_time'].mean()
        ax3.bar(method_time.index, method_time.values)
        ax3.set_ylabel('Average Processing Time (sec)')
        ax3.set_title('Processing Time by Method')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No processing time data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Processing Time')
    
    # Summary statistics table
    summary_text = f"""
GENERAL STATISTICS

Total Sheet Count: {len(sheets_summary)}
Processed with ML: {len(sheets_summary[sheets_summary['method'] == 'ML'])}
Processed with Simple Forecast: {len(sheets_summary[sheets_summary['method'] == 'Simple'])}

Total Row Count: {sheets_summary['row_count'].sum():,}
Average Rows/Sheet: {sheets_summary['row_count'].mean():.1f}
Largest Sheet: {sheets_summary['row_count'].max():,} rows
Smallest Sheet: {sheets_summary['row_count'].min():,} rows

Forecast Parameters:
- Horizon: {FORECAST_HORIZON_WEEKS} weeks
- ML Threshold: 100 rows
- XGBoost: {'Yes' if LIBRARY_AVAILABILITY['xgboost'] else 'No'}
- CatBoost: {'Yes' if LIBRARY_AVAILABILITY['catboost'] else 'No'}
- Prophet: {'Yes' if LIBRARY_AVAILABILITY['prophet'] else 'No'}

Rapor Oluşturma Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def _create_terminal_outputs_page(pdf: PdfPages):
    """Create the terminal outputs page"""
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.suptitle('Sistem Çıktıları ve Log Kayıtları', fontsize=16, fontweight='bold')
    
    # Get terminal contents
    terminal_content = get_terminal_contents()
    
    # Split into lines and format
    lines = terminal_content.split('\n')
    formatted_lines = []
    for i, line in enumerate(lines[-100:]):  # Last 100 lines
        if line.strip():
            formatted_lines.append(f"{i+1:3d}: {line}")
    
    terminal_text = '\n'.join(formatted_lines)
    
    ax.text(0.02, 0.98, terminal_text, transform=ax.transAxes, fontsize=8,
           verticalalignment='top', fontfamily='monospace', 
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def _create_detailed_summary_table(pdf: PdfPages, sheets_summary: pd.DataFrame):
    """Create detailed summary table page"""
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.suptitle('Detaylı Sayfa Analiz Tablosu', fontsize=16, fontweight='bold')
    
    # Create table
    display_columns = ['sheet_name', 'row_count', 'method']
    if 'forecast_mean' in sheets_summary.columns:
        display_columns.append('forecast_mean')
    if 'best_model' in sheets_summary.columns:
        display_columns.append('best_model')
    
    table_data = sheets_summary[display_columns].round(2)
    
    # Create column labels
    col_labels = ['Sayfa Adı', 'Satır Sayısı', 'Yöntem']
    if 'forecast_mean' in display_columns:
        col_labels.append('Ort. Tahmin')
    if 'best_model' in display_columns:
        col_labels.append('En İyi Model')
    
    # Create table visualization
    table = ax.table(cellText=table_data.values,
                   colLabels=col_labels,
                   cellLoc='center',
                   loc='center',
                   bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the table
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(color='white')
        else:
            if j == 2:  # Method column
                if cell.get_text().get_text() == 'ML':
                    cell.set_facecolor('#E3F2FD')
                else:
                    cell.set_facecolor('#FFF3E0')
            else:
                cell.set_facecolor('#F5F5F5')
    
    ax.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def _create_backtest_pages(pdf: PdfPages, all_forecasts: Dict, sheets_summary: pd.DataFrame):
    """Create individual backtest results pages"""
    for sheet_name, sheet_forecasts in all_forecasts.items():
        try:
            # Check if we have backtest results for this sheet
            has_backtest = False
            for model_name, forecast_df in sheet_forecasts.items():
                if hasattr(forecast_df, 'attrs') and forecast_df.attrs:
                    has_backtest = True
                    break
            
            if not has_backtest:
                continue
            
            # Create backtest results page
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Backtest Sonuçları - {sheet_name}', fontsize=16, fontweight='bold')
            
            # Model performance comparison
            ax1 = axes[0, 0]
            model_names = []
            mae_scores = []
            rmse_scores = []
            mape_scores = []
            wape_scores = []
            
            for model_name, forecast_df in sheet_forecasts.items():
                if hasattr(forecast_df, 'attrs') and forecast_df.attrs:
                    model_names.append(model_name.upper())
                    mae_scores.append(forecast_df.attrs.get('MAE', 0))
                    rmse_scores.append(forecast_df.attrs.get('RMSE', 0))
                    mape_scores.append(forecast_df.attrs.get('MAPE', 0))
                    wape_scores.append(forecast_df.attrs.get('WAPE', 0))
            
            if model_names:
                x = np.arange(len(model_names))
                width = 0.2
                
                ax1.bar(x - 1.5*width, mae_scores, width, label='MAE', alpha=0.8)
                ax1.bar(x - 0.5*width, rmse_scores, width, label='RMSE', alpha=0.8)
                ax1.bar(x + 0.5*width, mape_scores, width, label='MAPE', alpha=0.8)
                ax1.bar(x + 1.5*width, wape_scores, width, label='WAPE', alpha=0.8)
                
                ax1.set_xlabel('Modeller')
                ax1.set_ylabel('Hata Değeri')
                ax1.set_title('Model Performans Karşılaştırması')
                ax1.set_xticks(x)
                ax1.set_xticklabels(model_names)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Performance metrics table
            ax2 = axes[0, 1]
            ax2.axis('off')
            
            if model_names:
                # Create performance table
                table_data = []
                for i, model in enumerate(model_names):
                    table_data.append([
                        model,
                        f"{mae_scores[i]:.2f}",
                        f"{rmse_scores[i]:.2f}", 
                        f"{mape_scores[i]:.1f}%",
                        f"{wape_scores[i]:.1f}%",
                        "✅ HEDEF" if wape_scores[i] < 20 else "❌ HEDEF AŞILDI",
                        "🏆" if mae_scores[i] == min(mae_scores) else ""
                    ])
                
                table = ax2.table(cellText=table_data,
                               colLabels=['Model', 'MAE', 'RMSE', 'MAPE', 'WAPE', 'Hedef (%20)', 'En İyi'],
                               cellLoc='center',
                               loc='center',
                               bbox=[0, 0, 1, 1])
                
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.5)
                
                # Style table
                for (i, j), cell in table.get_celld().items():
                    if i == 0:  # Header
                        cell.set_text_props(weight='bold')
                        cell.set_facecolor('#2196F3')
                        cell.set_text_props(color='white')
                    else:
                        # Color by WAPE performance
                        if j == 5:  # WAPE target column
                            if "✅" in str(cell.get_text().get_text()):
                                cell.set_facecolor('#C8E6C9')
                            else:
                                cell.set_facecolor('#FFCDD2')
                        elif j == 6:  # Best model column
                            if "🏆" in str(cell.get_text().get_text()):
                                cell.set_facecolor('#FFF59D')
                        else:
                            cell.set_facecolor('#F5F5F5')
                
                ax2.set_title('Detaylı Performans Metrikleri', fontweight='bold', pad=20)
            
            # WAPE vs Target comparison
            ax3 = axes[1, 0]
            if wape_scores:
                colors = ['green' if score < 20 else 'red' for score in wape_scores]
                bars = ax3.bar(model_names, wape_scores, color=colors, alpha=0.7)
                ax3.axhline(y=20, color='red', linestyle='--', linewidth=2, label='Hedef (%20)')
                ax3.set_ylabel('WAPE (%)')
                ax3.set_title('WAPE vs Hedef Karşılaştırması')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, score in zip(bars, wape_scores):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # Summary stats
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # Find sheet info
            sheet_info = sheets_summary[sheets_summary['sheet_name'] == sheet_name]
            
            summary_text = f"📊 {sheet_name} - Özet\n\n"
            if not sheet_info.empty:
                summary_text += f"• Satır Sayısı: {sheet_info.iloc[0]['row_count']:,}\n"
                summary_text += f"• İşlem Yöntemi: {sheet_info.iloc[0]['method']}\n"
            
            if model_names:
                best_idx = mae_scores.index(min(mae_scores))
                summary_text += f"• En İyi Model: {model_names[best_idx]}\n"
                summary_text += f"• En Düşük WAPE: {min(wape_scores):.1f}%\n"
                summary_text += f"• Hedef Başarısı: {'✅ Başarılı' if min(wape_scores) < 20 else '❌ Hedef Aşıldı'}\n"
                summary_text += f"• Model Sayısı: {len(model_names)}\n\n"
                
                summary_text += "📈 Performans Sıralaması:\n"
                sorted_models = sorted(zip(model_names, wape_scores), key=lambda x: x[1])
                for i, (model, wape) in enumerate(sorted_models[:3]):
                    summary_text += f"{i+1}. {model}: {wape:.1f}%\n"
            
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            log_output(f"❌ {sheet_name} backtest sayfası oluşturulamadı: {e}")
            continue
