from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import optuna
import numpy as np
import pandas as pd
import warnings

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')


class NikkeiPredictor:
    def __init__(self):
        self.lr_model  = LinearRegression()
        self.lgb_model = None
        self.rf_model  = RandomForestRegressor(n_estimators=100, random_state=42)
        self.clf_model = None
        self.weights   = {"rf": 0.4, "lgb": 0.6}
        self.feature_cols = [
            'RSI', 'Psychological', 'Volume_Ret',
            'SMA_Dist5', 'SMA_Dist25', 'SMA_Dist75',
            'BB_Upper_Dist', 'BB_Lower_Dist',
            'MACD_Ratio', 'Signal_Ratio', 'ATR_Ratio',
            'Returns', 'Lag_1_Ret', 'Lag_2_Ret', 'Lag_3_Ret', 'Lag_4_Ret', 'Lag_5_Ret',
            'S&P500_Ret', 'DOW_Ret', 'NASDAQ_Ret',
            'USDJPY_Ret', 'VIX_Ret', 'TNX_Ret', 'Gold_Ret', 'Oil_Ret',
        ]

    # ─────────────────────────────────────────────
    def train(self, df, n_trials=30):
        """v10: 定常化特徴量によるアンサンブル学習"""
        X     = df[self.feature_cols]
        y     = df['Target']
        y_dir = (y > 0).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        y_train_dir = y_dir.iloc[:len(X_train)]

        print(f"Training v10 model on {len(X_train)} samples...")

        # 1. Linear Regression
        # 特徴量が完全に定常化されたため、直近4年(1000日)に広げて堅牢なトレンドベースラインを作る
        train_window = min(len(X_train), 1000)
        self.lr_model.fit(X_train.tail(train_window), y_train.tail(train_window))
        y_train_lr = self.lr_model.predict(X_train)
        y_test_lr  = self.lr_model.predict(X_test)
        residuals  = y_train - y_train_lr

        # 2. Random Forest（残差）
        self.rf_model.fit(X_train, residuals)

        # 3. LightGBM（残差） + Optuna
        def objective_lgb(trial):
            params = {
                'objective': 'regression', 'metric': 'rmse', 'verbosity': -1,
                'n_estimators':      trial.suggest_int('n_estimators', 100, 800),
                'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.05),
                'num_leaves':        trial.suggest_int('num_leaves', 16, 128),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            }
            tscv   = TimeSeriesSplit(n_splits=3)
            scores = []
            for t_idx, v_idx in tscv.split(X_train):
                m = lgb.LGBMRegressor(**params)
                m.fit(X_train.iloc[t_idx], residuals.iloc[t_idx])
                scores.append(np.sqrt(mean_squared_error(
                    residuals.iloc[v_idx], m.predict(X_train.iloc[v_idx])
                )))
            return np.mean(scores)

        study_lgb = optuna.create_study(direction='minimize')
        study_lgb.optimize(objective_lgb, n_trials=n_trials)
        self.lgb_model = lgb.LGBMRegressor(**study_lgb.best_params, verbosity=-1)
        self.lgb_model.fit(X_train, residuals)

        # 4. 方向性分類器
        print("Training Directional Classifier...")
        self.clf_model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.03, num_leaves=31, verbosity=-1
        )
        self.clf_model.fit(X_train, y_train_dir)

        # 5. アンサンブル比率最適化
        prf = self.rf_model.predict(X_test)
        plg = self.lgb_model.predict(X_test)

        def objective_w(trial):
            w = trial.suggest_float('w_rf', 0.0, 1.0)
            p = y_test_lr + (w * prf + (1 - w) * plg)
            return np.sqrt(mean_squared_error(y_test, p))

        study_w = optuna.create_study(direction='minimize')
        study_w.optimize(objective_w, n_trials=20)
        self.weights['rf']  = study_w.best_params['w_rf']
        self.weights['lgb'] = 1.0 - self.weights['rf']

        final = y_test_lr + (self.weights['rf'] * prf + self.weights['lgb'] * plg)
        return {"RMSE": np.sqrt(mean_squared_error(y_test, final))}

    # ─────────────────────────────────────────────
    def predict_target(self, latest_features):
        """v10: 符号整合補正付き予測"""
        X = latest_features[self.feature_cols]
        if len(X) > 1:
            X = X.iloc[-1:]

        pred_lr  = self.lr_model.predict(X)[0]
        pred_res = (self.weights['rf']  * self.rf_model.predict(X)[0]
                  + self.weights['lgb'] * self.lgb_model.predict(X)[0])
        base_pred = pred_lr + pred_res

        prob_up  = self.clf_model.predict_proba(X)[0][1]
        regr_up  = base_pred > 0
        clf_up   = prob_up >= 0.5
        clf_conf = abs(prob_up - 0.5) * 2

        if regr_up == clf_up:
            direction_sign = 1 if clf_up else -1
            scale = 1.0 + 0.2 * direction_sign * clf_conf
            return base_pred * max(0.8, min(1.2, scale))
        else:
            damping = 1.0 - clf_conf * 0.8
            return base_pred * damping

    # ─────────────────────────────────────────────
    def predict_next_day_range(self, df_all):
        latest = df_all.iloc[-1]
        close  = float(latest['Close'])

        # v10: ATRは正規化されたATR_Ratioから生ATRを逆算、あるいは生のHigh-Lowの移動平均
        # DataFrameには生のATRがもう無いので、計算し直すかRatioから復元
        atr = float(latest['ATR_Ratio']) * close

        vix = float(latest['VIX'])
        vix_factor = max(1.0, vix / 20.0)
        vol_1sigma = atr * 0.6 * vix_factor

        recent_ret = df_all['Returns'].tail(5).mean()
        bias = close * recent_ret
        center = close + bias

        return {
            "center":       round(center),
            "high_1sigma":  round(center + vol_1sigma),
            "low_1sigma":   round(center - vol_1sigma),
            "high_2sigma":  round(center + vol_1sigma * 2),
            "low_2sigma":   round(center - vol_1sigma * 2),
            "atr":          round(atr),
            "vix_factor":   round(vix_factor, 2),
        }


if __name__ == "__main__":
    pass

