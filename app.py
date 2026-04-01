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
    
    # テキストサマリーの生成
    st.divider()
    st.subheader("📝 AIによる市況サマリーレポート")
    
    vix_val = latest['VIX']
    if vix_val > 25:
        vix_alert = "非常に高くなっており、相場の急激な変動に最大級の警戒が必要です。"
    elif vix_val > 20:
        vix_alert = "やや高まっており、相場のボラティリティに警戒が必要です。"
    else:
        vix_alert = "落ち着いた水準にあり、比較的安定した値動きが予想されます。"
    
    trend_25d = results[-1]['トレンド']
    prob_25d = results[-1]['上昇確率']
    
    report_text = f"""本日の日経平均株価は **{current_price:,.2f} 円** となりました。

**【翌営業日の展望】**
市場のVIX（恐怖指数）は現在 {vix_val:.2f} と{vix_alert}
AIの予測モデルによれば、翌営業日は中心予想 **{next_range['center']:,.0f}円** を基準とし、
上値目途 **{next_range['high_2sigma']:,.0f}円** から 下値目途 **{next_range['low_2sigma']:,.0f}円** のレンジ内で推移するシナリオが有力視されています。

**【中長期の展望】**
約1ヶ月後（25営業日後）のトレンド予測では「**{trend_25d}**」（AIによる上昇確率: {prob_25d}）のサインが出ています。
足元の為替（USD/JPY: {latest['USDJPY']:.2f}）や米金利（TNX: {latest['TNX']:.2f}）の動向に注意を払いながら、ポジションの調整を図ることをお勧めします。

*(※本レポートはAIによる統計的確率予測であり、投資結果を保証するものではありません)*
"""
    st.info(report_text)
    
    with st.expander("テキストをコピーする（SNSやメモ用）"):
        st.code(report_text, language="markdown")

