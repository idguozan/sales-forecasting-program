# pipeline/forecast.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def forecast_future_sales(
    model,
    weekly: pd.DataFrame,
    all_products,
    forecast_horizon_weeks: int = 12,
):
    """
    RF rolling forecast loop from monolithic code (one-to-one).
    """
    all_products = list(all_products)
    future_records = []

    # Historical data dictionaries
    hist_qty_dict = {
        prod: weekly.loc[weekly['StockCode'] == prod, 'qty'].to_list()
        for prod in all_products
    }
    hist_price_dict = {
        prod: weekly.loc[weekly['StockCode'] == prod, 'avg_price'].iloc[-1]
        for prod in all_products
    }
    last_week_dict = {
        prod: weekly.loc[weekly['StockCode'] == prod, 'week'].max()
        for prod in all_products
    }

    print(f"Generating forecasts for the next {forecast_horizon_weeks} weeks (RF rolling)...")

    for idx, product in enumerate(all_products, start=1):
        qty_hist  = hist_qty_dict[product]
        last_price = hist_price_dict[product]
        last_week  = last_week_dict[product]

        future_qtys = []

        for _ in range(forecast_horizon_weeks):
            next_week = last_week + pd.Timedelta(weeks=1)

            history   = qty_hist + future_qtys
            qty_lag_1 = history[-1]
            qty_lag_4 = history[-4] if len(history) >= 4 else qty_lag_1
            qty_lag_8 = history[-8] if len(history) >= 8 else qty_lag_4

            month      = next_week.month
            week_no    = next_week.isocalendar().week
            promo_flag = 0  # simple approach

            X_future = pd.DataFrame({
                'avg_price': [last_price],
                'month': [month],
                'week_no': [week_no],
                'qty_lag_1': [qty_lag_1],
                'qty_lag_4': [qty_lag_4],
                'qty_lag_8': [qty_lag_8],
                'promo_flag': [promo_flag],
            })

            pred_qty = model.predict(X_future)[0]
            pred_qty = max(pred_qty, 0)

            future_records.append([product, next_week, pred_qty])
            future_qtys.append(pred_qty)
            last_week = next_week

        if idx % 10 == 0 or idx == len(all_products):
            print(f"  > {idx} / {len(all_products)} products processed...")

    future_df = pd.DataFrame(future_records, columns=['StockCode', 'week', 'predicted_qty'])
    return future_df


def plot_and_save_all(
    weekly: pd.DataFrame,
    future_df: pd.DataFrame,
    all_products,
    forecast_horizon_weeks: int,
    metrics: dict,
    plots_dir: str = "forecast_plots",
    total_plot_png: str = "total_weekly_forecast.png",
    pdf_report: str = "forecast_report.pdf",
):
    """
    As in the monolithic code:
    - Product PNGs
    - Total sales PNG
    - PDF: cover + metrics + total chart + top10 table + product pages
    """
    all_products = list(all_products)
    os.makedirs(plots_dir, exist_ok=True)

    # Total sales chart
    past_total = weekly.groupby('week')['qty'].sum().reset_index()
    future_total = future_df.groupby('week')['predicted_qty'].sum().reset_index()

    fig_tot, ax_tot = plt.subplots(figsize=(12, 5))
    ax_tot.plot(past_total['week'], past_total['qty'], label='Geçmiş Toplam Satış', marker='o')
    ax_tot.plot(future_total['week'], future_total['predicted_qty'], label='Gelecek Toplam Tahmin', linestyle='--', marker='x')
    ax_tot.set_title(f"Toplam Haftalık Satış (Geçmiş vs {forecast_horizon_weeks} Hafta Tahmin)")
    ax_tot.set_xlabel("Hafta")
    ax_tot.set_ylabel("Miktar")
    ax_tot.legend()
    fig_tot.tight_layout()
    fig_tot.savefig(total_plot_png)
    plt.close(fig_tot)
    print(f"Total weekly sales chart saved: {total_plot_png}")

    # Top10 products
    product_totals = (
        future_df.groupby('StockCode')['predicted_qty']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    top10 = product_totals.head(10)
    print("\n--- Top 10 Products with Highest Estimated Sales in Next 12 Weeks ---")
    print(top10.to_string(index=False))

    # PDF report
    print("Generating PDF report...")
    with PdfPages(pdf_report) as pdf:
        # Cover
        fig_cover, ax_cover = plt.subplots(figsize=(8.27, 11.69))
        ax_cover.axis('off')
        ax_cover.text(0.5, 0.8, "Sales Forecast Report", ha='center', va='center', fontsize=20)
        ax_cover.text(0.5, 0.7, "Data: Online Retail", ha='center', va='center', fontsize=12)
        ax_cover.text(0.5, 0.65, f"Forecast Horizon: {forecast_horizon_weeks} Weeks", ha='center', va='center', fontsize=12)
        pdf.savefig(fig_cover)
        plt.close(fig_cover)

        # Metrics page
        fig_met, ax_met = plt.subplots(figsize=(8.27, 11.69))
        ax_met.axis('off')
        metrics_text = (
            "--- Backtest Results ---\n"
            f"MAE : {metrics['mae']:.2f}\n"
            f"RMSE: {metrics['rmse']:.2f}\n"
            f"MAPE: {metrics['mape']:.2f}%\n"
        )
        ax_met.text(0.05, 0.95, metrics_text, va='top', fontsize=12, family='monospace')
        pdf.savefig(fig_met)
        plt.close(fig_met)

        # Total sales chart page
        fig_totpdf, ax_totpdf = plt.subplots(figsize=(10, 5))
        ax_totpdf.plot(past_total['week'], past_total['qty'], label='Geçmiş', marker='o')
        ax_totpdf.plot(future_total['week'], future_total['predicted_qty'], label='Tahmin', linestyle='--', marker='x')
        ax_totpdf.set_title(f"Toplam Haftalık Satış (Geçmiş vs {forecast_horizon_weeks} Hafta Tahmin)")
        ax_totpdf.set_xlabel("Hafta")
        ax_totpdf.set_ylabel("Miktar")
        ax_totpdf.legend()
        fig_totpdf.tight_layout()
        pdf.savefig(fig_totpdf)
        plt.close(fig_totpdf)

        # Top10 tablo sayfası
        fig_top10, ax_top10 = plt.subplots(figsize=(8.27, 11.69))
        ax_top10.axis('off')
        ax_top10.text(0.05, 0.95, f"Gelecek {forecast_horizon_weeks} Haftada Tahmini En Çok Satılacak 10 Ürün",
                      fontsize=14, va='top')
        ax_top10.text(0.05, 0.90, top10.to_string(index=False), family='monospace', fontsize=10, va='top')
        pdf.savefig(fig_top10)
        plt.close(fig_top10)

        # Ürün bazlı grafik sayfaları + PNG kaydetme
        print("Tüm ürün grafikleri kaydediliyor...")
        for idx, product in enumerate(all_products, start=1):
            past_data = weekly[weekly['StockCode'] == product]
            future_data = future_df[future_df['StockCode'] == product]
            if future_data.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(past_data['week'], past_data['qty'], label='Geçmiş Satış', marker='o')
            # Sadece tahmin noktalarını işaretle ve tarihleri göster
            ax.scatter(future_data['week'], future_data['predicted_qty'], color='orange', label='Tahmin Noktası', marker='x')
            for i, row in future_data.iterrows():
                ax.annotate(str(row['week'].date()), (row['week'], row['predicted_qty']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='orange')
            ax.set_title(f"Ürün {product} - Satış Tahmini")
            ax.set_xlabel("Tahmin Tarihi")
            ax.set_ylabel("Miktar")
            ax.legend()
            fig.tight_layout()

            # PNG kaydet (önce!)
            fig.savefig(os.path.join(plots_dir, f"{product}.png"))

            # PDF'e ekle
            pdf.savefig(fig)
            plt.close(fig)

    print(f"PNG grafikler '{plots_dir}/' klasörüne kaydedildi.")
    print(f"PDF raporu oluşturuldu: {pdf_report}")
