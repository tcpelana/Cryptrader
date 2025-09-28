# predictor.py
"""
predictor.py
- build_features(df, news_df=None): devuelve df con features listos
- train_and_predict(df): entrena modelo simple y devuelve DataFrame con predicciones
- If scikit-learn not available, se usa una regla simple (momentum)
"""

import pandas as pd
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

def build_features(df: pd.DataFrame, news_df: pd.DataFrame = None, ma_windows=[5, 10, 20]):
    d = df.copy().sort_index()
    d["ret1"] = d["close"].pct_change().fillna(0)
    for w in ma_windows:
        d[f"ma_{w}"] = d["close"].rolling(w).mean()
    d["vol_10"] = d["close"].pct_change().rolling(10).std().fillna(0)
    if news_df is not None:
        # align news daily to price index
        news_daily = news_df.reindex(pd.date_range(d.index.min().date(), d.index.max().date(), freq="D"), fill_value=0)
        news_daily.index = pd.to_datetime(news_daily.index)
        d["news_count"] = d.index.floor("D").map(lambda x: news_daily.loc[pd.to_datetime(x.date())]["news_count"] if pd.to_datetime(x.date()) in news_daily.index else 0)
        d["news_sent"] = d.index.floor("D").map(lambda x: news_daily.loc[pd.to_datetime(x.date())]["avg_sent"] if pd.to_datetime(x.date()) in news_daily.index else 0.0)
    else:
        d["news_count"] = 0
        d["news_sent"] = 0.0
    # drop NaNs
    d = d.fillna(method="bfill").fillna(0)
    return d

def train_and_predict(df: pd.DataFrame, feature_cols=None, target_shift=1):
    """
    Entrena un modelo para predecir dirección (up/down) del siguiente periodo.
    Retorna df con columnas: pred_proba_up, pred_label
    """
    d = df.copy().sort_index()
    if feature_cols is None:
        # choose sensible defaults
        feature_cols = [c for c in d.columns if c.startswith("ma_") or c in ("ret1", "vol_10", "news_sent", "news_count")]
    # build label: whether next close > current close
    d["label_next_up"] = (d["close"].shift(-target_shift) > d["close"]).astype(int)
    # drop last rows where label is NaN
    d = d.dropna(subset=["label_next_up"])
    if len(d) < 50:
        # not enough data: fallback naive rule
        d["pred_proba_up"] = 0.5 + d["ret1"].clip(-0.02, 0.02) * 10
        d["pred_label"] = (d["pred_proba_up"] > 0.5).astype(int)
        return d
    X = d[feature_cols].values
    y = d["label_next_up"].values
    if SKLEARN_AVAILABLE:
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X[:-10], y[:-10])  # train except last 10 for small validation
        proba = model.predict_proba(X)[:, 1]
        d["pred_proba_up"] = proba
        d["pred_label"] = (proba > 0.5).astype(int)
        # attach model if needed
        d.attrs["model"] = model
    else:
        # fallback: momentum-based probability
        d["pred_proba_up"] = 0.5 + d["ret1"].clip(-0.02, 0.02) * 10
        d["pred_label"] = (d["pred_proba_up"] > 0.5).astype(int)
    return d
