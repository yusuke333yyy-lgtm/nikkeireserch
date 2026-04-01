import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import sys
import os

from data_loader import get_multi_data, add_technical_indicators, prepare_dataset
from model import NikkeiPredictor

# Streamlit config
st.set_page_config(page_title="Nikkei 225 Forecast Dashboard", layout="wide", page_icon="📈")

st.title("📈 日経平均株価 マルチホライゾン予測ツール")
st.markdown("機械学習モデル（LightGBM）を用いて直近のデータから今後の日経平均株価を予測します。")

# 実行ボタン
if st.button("予測を実行する (Run Forecast)", type="primary"):
    with st.spinner("最新データを取得中（日経平均・S&P500・ドル円・金・原油・VIX・米金利）..."):
        try:
            df_raw = get_multi_data(period="10y")
        except Exception as e:
            st.error(f"データ取得エラー: {e}")
            st.stop()
            
    if df_raw.empty:
        st.error("データの取得に失敗しました。")
        st.stop()

    current_price = float(df_raw['Close'].iloc[-1])
    last_date = df_raw.index[-1]
    
    st.info(f"**基準日:** {last_date.strftime('%Y-%m-%d')}  ／  **現在値:** {current_price:,.2f} 円")
    
    with st.spinner("特徴量を生成中..."):
        df_all = add_technical_indicators(df_raw)
    
    # 翌営業日予想
    with st.spinner("翌営業日の予想レンジを推論・学習中 (Trials=10) ..."):
        df_train_1 = prepare_dataset(df_all, forecast_horizon=3)
        range_predictor = NikkeiPredictor()
        range_predictor.train(df_train_1, n_trials=10)  # 高速化のため10トライアル
        next_range = range_predictor.predict_next_day_range(df_all)
    
    st.divider()
    st.subheader("🎯 翌営業日 予想レンジ（ボラティリティ調整済）")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("▼ 下値目途 (-2σ)", f"{next_range['low_2sigma']:,.0f} 円", f"{next_range['low_2sigma'] - current_price:,.0f} 円", delta_color="inverse")
    col2.metric("🔹 中心予測", f"{next_range['center']:,.0f} 円", "")
    col3.metric("▲ 上値目途 (+2σ)", f"{next_range['high_2sigma']:,.0f} 円", f"+{next_range['high_2sigma'] - current_price:,.0f} 円", delta_color="normal")
    
    # マルチホライゾン予測
    st.divider()
    st.subheader("📆 マルチホライゾン予測（中長期トレンド）")
    
    horizons = [3, 5, 10, 25]
    results = []
    
    progress_text = "マルチホライゾンモデルを学習中..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, h in enumerate(horizons):
        my_bar.progress((i) / len(horizons), text=f"{h}営業日後の予測モデルを学習中...")
        df_train = prepare_dataset(df_all, forecast_horizon=h)
        predictor = NikkeiPredictor()
        predictor.train(df_train, n_trials=15)
        
        latest_row = df_all.iloc[-1:].copy()
        X_latest = latest_row[predictor.feature_cols]

        prob_up = predictor.clf_model.predict_proba(X_latest)[0][1]
        pred_return = predictor.predict_target(latest_row)
        pred_price = current_price * (1 + pred_return)
        
        # アイコン判定
        if pred_return > 0.01:
            direction = "🔺 力強い上昇"
        elif pred_return > 0:
            direction = "📈 やや上昇"
        elif pred_return < -0.01:
            direction = "🔻 強い下落警戒"
        else:
            direction = "📉 やや下落"
            
        results.append({
            "期間": f"{h}営業日後",
            "上昇確率": f"{prob_up:.1%}",
            "騰落率": f"{pred_return:+.2%}",
            "予測価格": f"{pred_price:,.0f} 円",
            "トレンド": direction
        })
        
    my_bar.progress(1.0, text="予測完了")
    my_bar.empty()
    
    res_df = pd.DataFrame(results)
    st.dataframe(res_df, use_container_width=True)
    
    # 指標一覧
    st.divider()
    st.subheader("💡 根拠となる主要指標の現在値")
    latest = df_all.iloc[-1]
    
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("VIX (恐怖指数)", f"{latest['VIX']:.2f}")
    m2.metric("TNX (米10年金利)", f"{latest['TNX']:.2f}")
    m3.metric("USD/JPY", f"{latest['USDJPY']:.2f}")
    m4.metric("Gold", f"{latest['Gold']:.1f}")
    m5.metric("Oil", f"{latest['Oil']:.2f}")
    m6.metric("SMA25 乖離率", f"{latest['SMA_Dist25']:.2%}")
    
    # チャート描画部分
    st.divider()
    st.subheader("📉 トレンド＆レンジ 予測チャート")
    with st.spinner("チャートを描画中..."):
        fig, axes = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]})
        
        # ─── 上段: マルチホライゾン予測チャート ───
        ax1 = axes[0]
        recent_df = df_raw.tail(60)
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
            ax1.scatter(future_date, pred_val, s=150, zorder=5)
            ax1.annotate(f"{h}d: {pred_val:,.0f}", (future_date, pred_val),
                         textcoords="offset points", xytext=(0, 15), ha='center', fontsize=10, fontweight='bold')

        ax1.plot(dates, prices, linestyle='--', color='#f44336', alpha=0.7, label='Forecast Trend')
        ax1.set_title(f'Nikkei 225 Multi-Horizon Forecast', fontsize=15)
        ax1.set_ylabel('Price (JPY)')
        ax1.grid(True, linestyle='--', alpha=0.4)
        ax1.legend()

        # ─── 下段: 翌営業日レンジ図 ───
        ax2 = axes[1]
        ax2.set_title('Next Trading Day Predicted Range', fontsize=13)
        ax2.set_xlim(0, 2)
        
        ax2.barh(0, next_range['high_2sigma'] - next_range['low_2sigma'], left=next_range['low_2sigma'], color='#BBDEFB', height=0.6, label='2σ Range (95%)')
        ax2.barh(0, next_range['high_1sigma'] - next_range['low_1sigma'], left=next_range['low_1sigma'], color='#64B5F6', height=0.6, label='1σ Range (68%)')
        ax2.axvline(x=next_range['center'], color='#1565C0', linewidth=3, linestyle='-', label=f"Center: {next_range['center']:,}")
        ax2.axvline(x=current_price, color='#FF7043', linewidth=3, linestyle='--', label=f"Current: {current_price:,.0f}")

        for val, label in [
            (next_range['low_2sigma'], f"▼2σ\n{next_range['low_2sigma']:,}"),
            (next_range['low_1sigma'], f"▼1σ\n{next_range['low_1sigma']:,}"),
            (next_range['high_1sigma'], f"▲1σ\n{next_range['high_1sigma']:,}"),
            (next_range['high_2sigma'], f"▲2σ\n{next_range['high_2sigma']:,}"),
        ]:
            ax2.text(val, 0.45, label, ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax2.set_yticks([])
        ax2.set_xlabel('Price (JPY)', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, axis='x', linestyle='--', alpha=0.4)

        plt.tight_layout()
        st.pyplot(fig)
