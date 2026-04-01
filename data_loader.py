import yfinance as yf
import pandas as pd
import numpy as np

def get_multi_data(period="10y"):
    """
    日経平均 (^N225), S&P500, NYダウ, NASDAQ, USD/JPY, VIX,
    米10年金利, 金, 原油 のデータを取得する。
    日経平均は複数手段で確実に取得する。
    """
    symbols = {
        "S&P500":  "^GSPC",
        "DOW":     "^DJI",    # ★ NYダウ追加
        "NASDAQ":  "^IXIC",   # ★ NASDAQ追加
        "USDJPY":  "JPY=X",
        "VIX":     "^VIX",
        "TNX":     "^TNX",
        "Gold":    "GC=F",
        "Oil":     "CL=F"
    }

    data_dict = {}
    for name, sym in symbols.items():
        print(f"Fetching {name} ({sym})...")
        try:
            df = yf.download(sym, period=period, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data_dict[name] = df['Close']
        except Exception as e:
            print(f"  WARNING: {name} の取得失敗 ({e})。スキップします。")

    # ── 日経平均は確実に取得する（3段階フォールバック） ──────────────
    print("Fetching N225 (^N225)...")
    n225_full = None

    # ① 通常取得
    try:
        n225_full = yf.download("^N225", period=period, progress=False, auto_adjust=True)
        if isinstance(n225_full.columns, pd.MultiIndex):
            n225_full.columns = n225_full.columns.get_level_values(0)
        if n225_full.empty:
            raise ValueError("空データ")
    except Exception as e:
        print(f"  WARNING: 通常取得失敗 ({e}). リトライ中...")

    # ② auto_adjust=False で再試行
    if n225_full is None or n225_full.empty:
        try:
            n225_full = yf.download("^N225", period=period, progress=False, auto_adjust=False)
            if isinstance(n225_full.columns, pd.MultiIndex):
                n225_full.columns = n225_full.columns.get_level_values(0)
        except Exception as e:
            print(f"  WARNING: 再試行も失敗 ({e}). start/end 指定で取得...")

    # ③ start/end 明示で取得
    if n225_full is None or n225_full.empty:
        try:
            import datetime
            end   = datetime.date.today()
            start = end - datetime.timedelta(days=int(period.replace("y","")) * 365 + 30)
            n225_full = yf.download("^N225", start=str(start), end=str(end), progress=False)
            if isinstance(n225_full.columns, pd.MultiIndex):
                n225_full.columns = n225_full.columns.get_level_values(0)
        except Exception as e:
            raise RuntimeError(f"日経平均データの取得に完全に失敗しました: {e}")

    # 最新の終値を確認
    latest_close = float(n225_full['Close'].iloc[-1])
    latest_date  = n225_full.index[-1].strftime('%Y-%m-%d')
    print(f"  N225 取得完了: {latest_date} 終値 {latest_close:,.0f} 円")

    # ── 統合 ──────────────────────────────────────────────────────
    combined = pd.DataFrame(data_dict)
    combined = combined.ffill()

    df_final = n225_full.copy()
    for col in data_dict.keys():
        if col in combined.columns:
            df_final[col] = combined[col]

    # N225のインデックスで再インデックスして前方補完
    df_final = df_final.ffill()

    return df_final


def add_technical_indicators(df):
    """テクニカル指標を追加する（すべて定常化：比率・乖離率）"""
    df = df.copy()
    close = df['Close']

    df['SMA_Dist5']  = (close - close.rolling(window=5).mean()) / close.rolling(window=5).mean()
    df['SMA_Dist25'] = (close - close.rolling(window=25).mean()) / close.rolling(window=25).mean()
    df['SMA_Dist75'] = (close - close.rolling(window=75).mean()) / close.rolling(window=75).mean()

    # RSI (そのまま0-100)
    delta = close.diff()
    gain  = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs    = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ボリンジャーバンド (乖離率)
    sma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    bb_upper = sma20 + (std20 * 2)
    bb_lower = sma20 - (std20 * 2)
    df['BB_Upper_Dist'] = (close - bb_upper) / bb_upper
    df['BB_Lower_Dist'] = (close - bb_lower) / bb_lower

    # MACD / Signal (終値に対する比率)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_Ratio']   = macd / close
    df['Signal_Ratio'] = signal / close

    # ATR (終値に対する比率)
    high_low = df['High'] - df['Low']
    atr = high_low.rolling(window=14).mean()
    df['ATR_Ratio'] = atr / close

    # 出来高 (変化率) - 0割り回避
    df['Volume_Ret'] = df['Volume'].pct_change()

    # Psychological (そのまま0-100)
    up_days = (delta > 0).rolling(window=12).sum()
    df['Psychological'] = (up_days / 12) * 100

    df['Returns'] = close.pct_change()

    # 外部データの変化率
    ext_cols = ["S&P500", "DOW", "NASDAQ", "USDJPY", "VIX", "TNX", "Gold", "Oil"]
    for col in ext_cols:
        if col in df.columns:
            df[f'{col}_Ret'] = df[col].pct_change()

    # 過去5営業日の「リターン」をラグ特徴量として追加
    for i in range(1, 6):
        df[f'Lag_{i}_Ret'] = df['Returns'].shift(i)

    # inf, -inf は欠損値として扱う
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    return df


def prepare_dataset(df, forecast_horizon=3):
    """学習用データセットを準備する。"""
    df = add_technical_indicators(df)
    future_close = df['Close'].shift(-forecast_horizon)
    df['Target'] = (future_close - df['Close']) / df['Close']
    df = df.dropna()
    return df



if __name__ == "__main__":
    df = get_multi_data()
    print(df.tail())
