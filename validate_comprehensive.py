#!/usr/bin/env python3
"""
Kapsamlı Tahmin Doğrulama Scripti - Miktar, Fiyat ve Ciro Analizi
Bu script, gerçek satış verileri ile modellerimizin tüm tahminlerini karşılaştırır.
"""

import pandas as pd
import os
from datetime import datetime

def load_actual_data_comprehensive():
    """Gerçek verileri kapsamlı olarak yükle - miktar, fiyat, ciro"""
    actual_data = [
        # 2024-12-16 verileri
        {'date': '2024-12-16', 'product': 'Laptop', 'actual_qty': 52, 'actual_price': 12750.0, 'actual_revenue': 663000.0},
        {'date': '2024-12-16', 'product': 'Tablet', 'actual_qty': 101, 'actual_price': 7000.0, 'actual_revenue': 707000.0},
        {'date': '2024-12-16', 'product': 'Kulaklık', 'actual_qty': 88, 'actual_price': 1350.0, 'actual_revenue': 118800.0},
        {'date': '2024-12-16', 'product': 'Akıllı Saat', 'actual_qty': 78, 'actual_price': 3150.0, 'actual_revenue': 245700.0},
        {'date': '2024-12-16', 'product': 'Telefon', 'actual_qty': 85, 'actual_price': 9000.0, 'actual_revenue': 765000.0},
        {'date': '2024-12-16', 'product': 'Monitör', 'actual_qty': 78, 'actual_price': 4050.0, 'actual_revenue': 315900.0},
        
        # 2024-12-23 verileri
        {'date': '2024-12-23', 'product': 'Laptop', 'actual_qty': 107, 'actual_price': 15000.0, 'actual_revenue': 1605000.0},
        {'date': '2024-12-23', 'product': 'Tablet', 'actual_qty': 96, 'actual_price': 5950.0, 'actual_revenue': 571200.0},
        {'date': '2024-12-23', 'product': 'Kulaklık', 'actual_qty': 87, 'actual_price': 1500.0, 'actual_revenue': 130500.0},
        {'date': '2024-12-23', 'product': 'Akıllı Saat', 'actual_qty': 107, 'actual_price': 3325.0, 'actual_revenue': 355775.0},
        {'date': '2024-12-23', 'product': 'Telefon', 'actual_qty': 76, 'actual_price': 8500.0, 'actual_revenue': 646000.0},
        {'date': '2024-12-23', 'product': 'Monitör', 'actual_qty': 18, 'actual_price': 4050.0, 'actual_revenue': 72900.0},
    ]
    
    return pd.DataFrame(actual_data)

def load_historical_prices():
    """Geçmiş verilerden ortalama fiyatları çıkar"""
    try:
        df = pd.read_csv('uploads/haftalik_satis_verisi.csv')
        price_averages = df.groupby('Urun')['Fiyat'].mean().to_dict()
        return price_averages
    except:
        return {}

def load_predictions_comprehensive():
    """Tüm model tahminlerini yükle"""
    models = ['rf', 'et', 'gb', 'xgb', 'cat', 'improved']
    predictions = {}
    
    for model in models:
        file_path = f'app/static/reports/forecast_{model}.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # Dosya formatını kontrol et
            if 'week' in df.columns and df['week'].dtype == 'object':
                # Tarih formatındaysa (rf, et, gb, xgb, cat modelleri)
                df['date'] = df['week']
                df['product'] = df['sheet_name']
                df['predicted_qty'] = df['forecast']
                predictions[model] = df[['date', 'product', 'predicted_qty']].copy()
            elif 'week' in df.columns and 'sheet_name' in df.columns:
                # Improved model formatı (hafta numarasıyla)
                base_date = pd.to_datetime('2024-12-16')
                df['date'] = df['week'].apply(lambda x: (base_date + pd.Timedelta(weeks=x-1)).strftime('%Y-%m-%d'))
                df['product'] = df['sheet_name']
                df['predicted_qty'] = df['forecast']
                predictions[model] = df[['date', 'product', 'predicted_qty']].copy()
    
    return predictions

def calculate_comprehensive_metrics(actual_qty, predicted_qty, actual_price, avg_price, actual_revenue, predicted_revenue):
    """Kapsamlı doğruluk metriklerini hesapla"""
    
    # Miktar metrikleri
    qty_error = abs(actual_qty - predicted_qty)
    qty_percentage_error = (qty_error / actual_qty) * 100 if actual_qty != 0 else float('inf')
    qty_accuracy = max(0, 100 - qty_percentage_error)
    
    # Fiyat metrikleri
    price_error = abs(actual_price - avg_price) if avg_price else 0
    price_percentage_error = (price_error / actual_price) * 100 if actual_price != 0 and avg_price else 0
    price_accuracy = max(0, 100 - price_percentage_error) if avg_price else 0
    
    # Ciro metrikleri
    revenue_error = abs(actual_revenue - predicted_revenue)
    revenue_percentage_error = (revenue_error / actual_revenue) * 100 if actual_revenue != 0 else float('inf')
    revenue_accuracy = max(0, 100 - revenue_percentage_error)
    
    return {
        'qty_error': qty_error,
        'qty_percentage_error': qty_percentage_error,
        'qty_accuracy': qty_accuracy,
        'price_error': price_error,
        'price_percentage_error': price_percentage_error,
        'price_accuracy': price_accuracy,
        'revenue_error': revenue_error,
        'revenue_percentage_error': revenue_percentage_error,
        'revenue_accuracy': revenue_accuracy
    }

def validate_comprehensive():
    """Ana kapsamlı doğrulama fonksiyonu"""
    print("🔍 Kapsamlı Tahmin Doğrulama Başlatılıyor...")
    print("📊 Miktar + Fiyat + Ciro Analizi")
    print("=" * 70)
    
    # Verileri yükle
    actual_df = load_actual_data_comprehensive()
    predictions = load_predictions_comprehensive()
    price_averages = load_historical_prices()
    
    if not predictions:
        print("❌ Tahmin dosyaları bulunamadı!")
        return
    
    print(f"📈 Geçmiş verilerden ortalama fiyatlar yüklendi: {len(price_averages)} ürün")
    print()
    
    # Her model için kapsamlı karşılaştırma
    all_results = []
    
    for model_name, pred_df in predictions.items():
        print(f"\n📊 {model_name.upper()} Modeli Kapsamlı Değerlendirmesi:")
        print("-" * 60)
        
        model_results = []
        
        for _, actual_row in actual_df.iterrows():
            date = actual_row['date']
            product = actual_row['product']
            actual_qty = actual_row['actual_qty']
            actual_price = actual_row['actual_price']
            actual_revenue = actual_row['actual_revenue']
            
            # Bu tarih ve ürün için tahmin bul
            pred_row = pred_df[
                (pred_df['date'] == date) & 
                (pred_df['product'] == product)
            ]
            
            if not pred_row.empty:
                predicted_qty = pred_row.iloc[0]['predicted_qty']
                avg_price = price_averages.get(product, actual_price)
                predicted_revenue = predicted_qty * avg_price
                
                metrics = calculate_comprehensive_metrics(
                    actual_qty, predicted_qty, actual_price, avg_price, 
                    actual_revenue, predicted_revenue
                )
                
                print(f"   📦 {product} ({date}):")
                print(f"     📊 Miktar  - Gerçek: {actual_qty:3d} | Tahmin: {predicted_qty:6.1f} | Doğruluk: {metrics['qty_accuracy']:5.1f}%")
                print(f"     💰 Fiyat  - Gerçek: {actual_price:8.0f} TL | Ort: {avg_price:8.0f} TL | Doğruluk: {metrics['price_accuracy']:5.1f}%")
                print(f"     💵 Ciro   - Gerçek: {actual_revenue:10.0f} | Tahmin: {predicted_revenue:10.0f} | Doğruluk: {metrics['revenue_accuracy']:5.1f}%")
                print()
                
                model_results.append({
                    'model': model_name,
                    'date': date,
                    'product': product,
                    'actual_qty': actual_qty,
                    'predicted_qty': predicted_qty,
                    'actual_price': actual_price,
                    'avg_price': avg_price,
                    'actual_revenue': actual_revenue,
                    'predicted_revenue': predicted_revenue,
                    **metrics
                })
        
        # Model ortalama performansı
        if model_results:
            avg_qty_acc = sum(r['qty_accuracy'] for r in model_results) / len(model_results)
            avg_price_acc = sum(r['price_accuracy'] for r in model_results) / len(model_results)
            avg_revenue_acc = sum(r['revenue_accuracy'] for r in model_results) / len(model_results)
            
            print(f"   🎯 {model_name.upper()} Ortalama Performans:")
            print(f"     📊 Miktar Doğruluğu: {avg_qty_acc:.1f}%")
            print(f"     💰 Fiyat Doğruluğu: {avg_price_acc:.1f}%")
            print(f"     💵 Ciro Doğruluğu: {avg_revenue_acc:.1f}%")
            print(f"     🏆 Genel Ortalama: {(avg_qty_acc + avg_price_acc + avg_revenue_acc)/3:.1f}%")
        
        all_results.extend(model_results)
    
    # Kapsamlı özet
    if all_results:
        print("\n" + "=" * 70)
        print("📈 KAPSAMLI PERFORMANS ÖZETİ")
        print("=" * 70)
        
        # Model bazında kapsamlı özet
        models_summary = {}
        for result in all_results:
            model = result['model']
            if model not in models_summary:
                models_summary[model] = {'qty': [], 'price': [], 'revenue': []}
            models_summary[model]['qty'].append(result['qty_accuracy'])
            models_summary[model]['price'].append(result['price_accuracy'])
            models_summary[model]['revenue'].append(result['revenue_accuracy'])
        
        print("\n🏆 Model Sıralaması (Kapsamlı Performansa Göre):")
        model_averages = []
        for model, data in models_summary.items():
            avg_qty = sum(data['qty']) / len(data['qty'])
            avg_price = sum(data['price']) / len(data['price'])
            avg_revenue = sum(data['revenue']) / len(data['revenue'])
            overall_avg = (avg_qty + avg_price + avg_revenue) / 3
            model_averages.append((model, avg_qty, avg_price, avg_revenue, overall_avg))
        
        model_averages.sort(key=lambda x: x[4], reverse=True)
        
        for i, (model, qty_acc, price_acc, revenue_acc, overall) in enumerate(model_averages, 1):
            print(f"   {i}. {model.upper()}:")
            print(f"      📊 Miktar: {qty_acc:.1f}% | 💰 Fiyat: {price_acc:.1f}% | 💵 Ciro: {revenue_acc:.1f}%")
            print(f"      🏆 Genel: {overall:.1f}%")
        
        # Ürün bazında en iyi performans
        print("\n📦 Ürün Bazında En İyi Ciro Tahmin Performansı:")
        products = set(r['product'] for r in all_results)
        
        for product in sorted(products):
            product_results = [r for r in all_results if r['product'] == product]
            if product_results:
                best_result = max(product_results, key=lambda x: x['revenue_accuracy'])
                print(f"   {product}: {best_result['model'].upper()} "
                      f"(Ciro: {best_result['revenue_accuracy']:.1f}%)")
        
        # En büyük ciro hataları
        print("\n⚠️  En Büyük Ciro Tahmin Hataları:")
        revenue_errors = sorted(all_results, key=lambda x: x['revenue_percentage_error'], reverse=True)[:5]
        
        for result in revenue_errors:
            print(f"   {result['product']} ({result['date']}) - {result['model'].upper()}:")
            print(f"     Gerçek: {result['actual_revenue']:,.0f} TL | Tahmin: {result['predicted_revenue']:,.0f} TL")
            print(f"     Hata: {result['revenue_percentage_error']:.1f}%")
    
    # Sonuçları CSV olarak kaydet
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_file = 'comprehensive_validation_results.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Kapsamlı sonuçlar kaydedildi: {output_file}")

if __name__ == "__main__":
    validate_comprehensive()
