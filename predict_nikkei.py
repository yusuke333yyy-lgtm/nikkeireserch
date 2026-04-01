import sys
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Headless mode
import pandas as pd
import numpy as np
from datetime import timedelta
from data_loader import get_multi_data, prepare_dataset, add_technical_indicators
from model import NikkeiPredictor

def run_prediction():
    print("=== 日経平均株価 マルチホライゾン予測ツール (v9) ===")
    
    # 1. データ取得
    print("最新データ（日経平均・S&P500・ドル円・金・原油・VIX・米金利）を取得中...")
    try:
        df_raw = get_multi_data(period="10y")
        if df_raw.empty:
            print("エラー: データの取得に失敗しました。")
            return
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return

    horizons = [3, 5, 10, 25]
    current_price = float(df_raw['Close'].iloc[-1])
    last_date = df_raw.index[-1]
    
    print(f"\n基準日: {last_date.strftime('%Y-%m-%d')}")
    print(f"現在値: {current_price:,.2f} 円")
    print("-" * 40)

    df_all = add_technical_indicators(df_raw)

    # ─── ① 翌営業日の予想レンジ（新機能） ───
    print("\n【翌営業日の予想レンジ】を算出中（3日後モデルを基に学習）...")
    df_train_1 = prepare_dataset(df_all, forecast_horizon=3)
    range_predictor = NikkeiPredictor()
    range_predictor.train(df_train_1, n_trials=20)
    next_range = range_predictor.predict_next_day_range(df_all)

    print("\n" + "="*65)
    print(f"翌営業日の予想レンジ (基準日: {last_date.strftime('%Y-%m-%d')})")
    print("="*65)
    print(f"  中心予測   : {next_range['center']:>8,} 円")
    print(f"  【強気シナリオ (2σ)】高値目途: {next_range['high_2sigma']:>8,} 円  (+{next_range['high_2sigma']-current_price:,.0f} 円)")
    print(f"  【強気シナリオ (1σ)】高値目途: {next_range['high_1sigma']:>8,} 円  (+{next_range['high_1sigma']-current_price:,.0f} 円)")
    print(f"  ────────────┤ 現在値 {current_price:>8,.0f} 円 ├────────────")
    print(f"  【弱気シナリオ (1σ)】安値目途: {next_range['low_1sigma']:>8,} 円  ({next_range['low_1sigma']-current_price:,.0f} 円)")
    print(f"  【弱気シナリオ (2σ)】安値目途: {next_range['low_2sigma']:>8,} 円  ({next_range['low_2sigma']-current_price:,.0f} 円)")
    print(f"\n  ※ ATR: {next_range['atr']:,} 円  VIX調整係数: ×{next_range['vix_factor']}")
    print("="*65)

    # ─── ② マルチホライゾン予測 ───
    results = []
    for h in horizons:
        print(f"\n【{h}営業日後】のモデルを学習・予測中...")
        df_train = prepare_dataset(df_all, forecast_horizon=h)
        
        predictor = NikkeiPredictor()
        predictor.train(df_train, n_trials=20)
        
        latest_row = df_all.iloc[-1:].copy()
        X_latest = latest_row[predictor.feature_cols]

        prob_up = predictor.clf_model.predict_proba(X_latest)[0][1]
        pred_return = predictor.predict_target(latest_row)
        pred_price = current_price * (1 + pred_return)
        direction = "▲上昇" if pred_return > 0 else "▼下落"
        
        results.append({
            "予測期間": f"{h}日後",
            "上昇確率": f"{prob_up:.1%}",
            "予測騰落率": f"{pred_return:+.2%}",
            "予測価格": f"{pred_price:,.0f} 円",
            "トレンド": direction
        })

    # ─── ③ 結果の出力 ───
    print("\n" + "="*65)
    print(f"マルチホライゾン予測レポート (基準日: {last_date.strftime('%Y-%m-%d')})")
    print("="*65)
    import tabulate
    res_df = pd.DataFrame(results)
    print(tabulate.tabulate(res_df, headers='keys', tablefmt='grid', showindex=False))
    print("="*65)
    
    print("\n【根拠となる主要指標の現在値】")
    latest = df_all.iloc[-1]
    indicators = {
        "VIX (恐怖指数)": latest['VIX'],
        "TNX (米10年金利)": latest['TNX'],
        "USDJPY (為替)": latest['USDJPY'],
        "Gold (金)": latest['Gold'],
        "Oil (原油)": latest['Oil'],
        "SMA25乖離率": f"{latest['SMA_Dist25']:.2%}"
    }
    for k, v in indicators.items():
        print(f" - {k}: {v}")


    # ─── ④ チャート描画（翌日レンジ + マルチホライゾン） ───
    plot_full_results(df_raw, res_df, last_date, horizons, next_range, current_price)


def plot_full_results(df, res_df, last_date, horizons, next_range, current_price):
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # ─── 上段: マルチホライゾン予測チャート ───
    ax1 = axes[0]
    recent_df = df.tail(60)
    ax1.plot(recent_df.index, recent_df['Close'], label='Actual Price', color='#2196F3', linewidth=2)
    
    from pandas.tseries.offsets import CustomBusinessDay
    bday = CustomBusinessDay()
    
    prices = [current_price]
    dates  = [last_date]
    
    for i, h in enumerate(horizons):
        future_date = last_date + (bday * h)
        pred_val = float(res_df.iloc[i]['予測価格'].replace(' 円', '').replace(',', ''))
        prices.append(pred_val)
        dates.append(future_date)
        ax1.scatter(future_date, pred_val, s=120, zorder=5)
        ax1.annotate(f"{h}d: {pred_val:,.0f}", (future_date, pred_val),
                     textcoords="offset points", xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')

    ax1.plot(dates, prices, linestyle='--', color='#f44336', alpha=0.7, label='Forecast Trend')
    ax1.set_title(f'Nikkei 225 Multi-Horizon Forecast (as of {last_date.strftime("%Y-%m-%d")})', fontsize=13)
    ax1.set_ylabel('Price (JPY)')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend()

    # ─── 下段: 翌営業日レンジ図 ───
    ax2 = axes[1]
    ax2.set_title('Next Trading Day Predicted Range', fontsize=12)
    ax2.set_xlim(0, 2)
    
    # 2σ帯（薄い）
    ax2.barh(0, next_range['high_2sigma'] - next_range['low_2sigma'],
             left=next_range['low_2sigma'], color='#BBDEFB', height=0.6, label='2σ Range (95%)')
    # 1σ帯（濃い）
    ax2.barh(0, next_range['high_1sigma'] - next_range['low_1sigma'],
             left=next_range['low_1sigma'], color='#64B5F6', height=0.6, label='1σ Range (68%)')
    # 中心値
    ax2.axvline(x=next_range['center'], color='#1565C0', linewidth=2, linestyle='-', label=f"Center: {next_range['center']:,}")
    # 現在値
    ax2.axvline(x=current_price, color='#FF7043', linewidth=2, linestyle='--', label=f"Current: {current_price:,.0f}")

    # ラベル
    for val, label in [
        (next_range['low_2sigma'], f"▼2σ\n{next_range['low_2sigma']:,}"),
        (next_range['low_1sigma'], f"▼1σ\n{next_range['low_1sigma']:,}"),
        (next_range['high_1sigma'], f"▲1σ\n{next_range['high_1sigma']:,}"),
        (next_range['high_2sigma'], f"▲2σ\n{next_range['high_2sigma']:,}"),
    ]:
        ax2.text(val, 0.42, label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_yticks([])
    ax2.set_xlabel('Price (JPY)')
    ax2.legend(loc='upper right')
    ax2.grid(True, axis='x', linestyle='--', alpha=0.4)

    plt.tight_layout()
    save_path = "prediction_chart_multi.png"
    plt.savefig(save_path, dpi=100)
    print(f"\n予測チャートを {save_path} に保存しました。")


if __name__ == "__main__":
    run_prediction()
