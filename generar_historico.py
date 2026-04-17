"""
generar_historico.py
====================
Traduccion 1-a-1 del notebook comparativa_e6_e11.ipynb + generar_trades_historicos.ipynb.
Entrena el HMM sobre datos diarios, aplica el pipeline y simula E6/E11 sobre el test set.

Se ejecuta periodicamente en GitHub Actions y publica docs/historical_trades.json.

Dependencias: pip install yfinance pandas numpy hmmlearn xgboost scikit-learn pandas_ta requests
"""

import json
import os
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

MODELS_DIR = Path("models")
OUT_FILE   = Path("docs/historical_trades.json")

# ── Parametros identicos a comparativa_e6_e11.ipynb ──────────────────────────
COMISION     = 0.0004
SLIPPAGE     = 0.0005
CAPITAL_INI  = 57.94
TRAIN_PCT    = 0.70
VAL_PCT      = 0.85

# E6
E6_DELTA = 0.30
E6_APAL  = 5
E6_PCT   = 0.40

# E11
E11_DELTA     = 0.20
E11_APAL      = 5
E11_BASE_PCT  = 0.20
E11_PCT_MAX   = 0.55
E11_KELLY_DIV = 3.0

VOL_UMBRAL = 0.80

FEATURE_COLS = [
    "log_ret_1h", "log_ret_4h", "log_ret_1d",
    "log_ret_eth", "log_ret_gold", "log_ret_dxy",
    "ema21_diff", "ema50_diff", "sma200_diff",
    "macd", "macd_signal", "macd_hist",
    "rsi_14", "roc_6", "roc_12", "roc_24", "mom_12",
    "atr_pct", "bb_width", "bb_pct", "vol_roll_24h",
    "obv_norm", "volume_zscore",
    "rsi_eth", "corr_btc_eth_24h", "fear_greed_norm",
]


# ═════════════════════════════════════════════════════════════════════════════
# 1. Descarga de datos (via yfinance, sustituye los CSV del Drive)
# ═════════════════════════════════════════════════════════════════════════════
def descargar_yf(ticker, periodo, intervalo):
    print(f"[YF] Descargando {ticker} ({periodo}, {intervalo})...")
    df = yf.download(ticker, period=periodo, interval=intervalo,
                     auto_adjust=False, progress=False)
    if df.empty:
        print(f"[YF] VACIO para {ticker}")
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.index.name = "datetime"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def descargar_todo():
    # yfinance limita 1h a 730 dias. Descargamos el maximo disponible.
    df_btc  = descargar_yf("BTC-USD",  "720d", "1h")
    df_eth  = descargar_yf("ETH-USD",  "720d", "1h")
    df_gold = descargar_yf("GC=F",     "2y",   "1d")
    df_dxy  = descargar_yf("DX-Y.NYB", "2y",   "1d")

    df_btc = df_btc[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    df_eth = df_eth[["close"]].rename(columns={"close": "eth"})
    df_gold = df_gold[["close"]].rename(columns={"close": "gold"})
    df_dxy  = df_dxy[["close"]].rename(columns={"close": "dxy"})

    # Fear & Greed
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=2000&format=json", timeout=10)
        dat = r.json()["data"]
        fg = pd.DataFrame(dat)[["timestamp", "value"]]
        fg["timestamp"] = pd.to_datetime(fg["timestamp"].astype(int), unit="s")
        fg = fg.set_index("timestamp").sort_index()
        fg["value"] = fg["value"].astype(float)
        fg.index.name = "datetime"
    except Exception as e:
        print(f"[FNG] Error: {e}. Usando valor neutro 50.")
        fg = pd.DataFrame({"value": [50]}, index=[df_btc.index[0]])

    return df_btc, df_eth, df_gold, df_dxy, fg


# ═════════════════════════════════════════════════════════════════════════════
# 2. Feature engineering — traduccion exacta del notebook
# ═════════════════════════════════════════════════════════════════════════════
def construir_features(df_btc, df_eth, df_gold, df_dxy, fg):
    print("[FEAT] Calculando features...")
    df = df_btc[["open", "high", "low", "close", "volume"]].copy()

    for asset in [df_eth, df_gold, df_dxy]:
        df = df.join(asset.reindex(df.index, method="ffill"), how="left")

    df["fear_greed"] = fg.reindex(df.index, method="ffill")["value"]
    df["log_ret_1h"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna(subset=["close", "log_ret_1h"])

    df["log_ret_4h"]  = df["log_ret_1h"].rolling(4).sum()
    df["log_ret_1d"]  = df["log_ret_1h"].rolling(24).sum()
    df["log_ret_eth"] = np.log(df["eth"] / df["eth"].shift(1))
    df["log_ret_gold"] = np.log(df["gold"] / df["gold"].shift(1)).ffill()
    df["log_ret_dxy"]  = np.log(df["dxy"]  / df["dxy"].shift(1)).ffill()

    close = df["close"]
    ema21  = ta.ema(close, length=21)
    ema50  = ta.ema(close, length=50)
    sma200 = ta.sma(close, length=200)
    df["ema21_diff"]  = (close - ema21)  / close
    df["ema50_diff"]  = (close - ema50)  / close
    df["sma200_diff"] = (close - sma200) / close

    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    df["macd"]        = macd_df["MACD_12_26_9"]  / close
    df["macd_signal"] = macd_df["MACDs_12_26_9"] / close
    df["macd_hist"]   = macd_df["MACDh_12_26_9"] / close

    df["rsi_14"] = ta.rsi(close, length=14)
    df["roc_6"]  = ta.roc(close, length=6)
    df["roc_12"] = ta.roc(close, length=12)
    df["roc_24"] = ta.roc(close, length=24)
    df["mom_12"] = ta.mom(close, length=12) / close

    atr_v = ta.atr(df["high"], df["low"], close, length=14)
    df["atr_pct"] = atr_v / close

    bb = ta.bbands(close, length=20, std=2)
    bb_u = [c for c in bb.columns if c.startswith("BBU")][0]
    bb_l = [c for c in bb.columns if c.startswith("BBL")][0]
    bb_m = [c for c in bb.columns if c.startswith("BBM")][0]
    df["bb_width"] = (bb[bb_u] - bb[bb_l]) / bb[bb_m]
    df["bb_pct"]   = (close - bb[bb_l]) / (bb[bb_u] - bb[bb_l])

    df["vol_roll_24h"] = df["log_ret_1h"].rolling(24).std()

    obv_v = ta.obv(close, df["volume"])
    df["obv_norm"] = (obv_v - obv_v.rolling(24).mean()) / (obv_v.rolling(24).std() + 1e-9)

    vm = df["volume"].rolling(24).mean()
    vs = df["volume"].rolling(24).std()
    df["volume_zscore"] = (df["volume"] - vm) / (vs + 1e-9)

    df["rsi_eth"] = ta.rsi(df["eth"], length=14)
    df["corr_btc_eth_24h"] = df["log_ret_1h"].rolling(24).corr(df["log_ret_eth"])

    df["fear_greed_norm"] = df["fear_greed"] / 100.0

    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.iloc[:-1]

    for col in ["gold", "dxy", "log_ret_gold", "log_ret_dxy"]:
        if col in df.columns:
            df[col] = df[col].ffill()

    df = df.dropna(subset=FEATURE_COLS + ["target"])
    print(f"[FEAT] Shape final: {df.shape}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 3. HMM sobre datos diarios — traduccion exacta del notebook
# ═════════════════════════════════════════════════════════════════════════════
def regimen_hmm(df, df_btc):
    print("[HMM] Resampleando a diario y entrenando...")

    df_btc_1d = df_btc.resample("1D").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["close"])

    df_daily = df_btc_1d.copy()
    df_daily["log_ret"]      = np.log(df_daily["close"] / df_daily["close"].shift(1))
    df_daily["vol_roll_20d"] = df_daily["log_ret"].rolling(20).std()
    vm_d = df_daily["volume"].rolling(20).mean()
    vs_d = df_daily["volume"].rolling(20).std()
    df_daily["volume_zscore"] = (df_daily["volume"] - vm_d) / (vs_d + 1e-9)
    df_daily = df_daily.dropna()

    X_hmm = df_daily[["log_ret", "vol_roll_20d", "volume_zscore"]].copy()
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_hmm)

    # Best-of-20 para robustez (critico para regimenes coherentes)
    best_m, best_s = None, -np.inf
    print("[HMM] Entrenando 20 HMM con seeds distintas...")
    for i in range(20):
        m = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=2000,
            tol=1e-5,
            random_state=i,
        )
        try:
            m.fit(X_sc)
            s = m.score(X_sc)
            if s > best_s:
                best_s = s
                best_m = m
        except Exception as e:
            print(f"[HMM] seed {i} fallo: {e}")

    print(f"[HMM] Mejor log-likelihood: {best_s:.2f}")

    states = best_m.predict(X_sc)
    df_daily["state_raw"] = states

    # Mapear los 3 estados a BULL / SIDEWAYS / BEAR segun media de retorno
    mean_ret = df_daily.groupby("state_raw")["log_ret"].mean()
    sorted_s = mean_ret.sort_values().index.tolist()
    state_map = {sorted_s[0]: "BEAR", sorted_s[1]: "SIDEWAYS", sorted_s[2]: "BULL"}
    df_daily["regime"] = df_daily["state_raw"].map(state_map)

    print(f"[HMM] Distribucion: {df_daily['regime'].value_counts().to_dict()}")

    # Expandir a frecuencia 1h via forward-fill por dia
    reg_d = df_daily[["regime"]].copy()
    reg_d.index = pd.to_datetime(reg_d.index).normalize()
    rm = reg_d.reindex(df.index.normalize(), method="ffill")
    rm.index = df.index
    df["regime"] = rm["regime"].values

    # Vol percentile expanding (sin leakage)
    df_daily["vol_percentile"] = df_daily["vol_roll_20d"].expanding().rank(pct=True)
    vp_d = df_daily[["vol_percentile"]].copy()
    vp_d.index = pd.to_datetime(vp_d.index).normalize()
    vp_m = vp_d.reindex(df.index.normalize(), method="ffill")
    vp_m.index = df.index
    df["vol_percentile"] = vp_m["vol_percentile"].values

    # EMA200 slope (5 dias de diferencia)
    ema200 = df_daily["close"].ewm(span=200, adjust=False).mean()
    ema200_slope = ema200 - ema200.shift(5)
    es_d = pd.DataFrame({"ema_slope": ema200_slope})
    es_d.index = pd.to_datetime(es_d.index).normalize()
    es_m = es_d.reindex(df.index.normalize(), method="ffill")
    es_m.index = df.index
    df["ema200_slope"] = es_m["ema_slope"].values

    return df, df_daily


# ═════════════════════════════════════════════════════════════════════════════
# 4. Split + probabilidades sobre el test set — traduccion exacta
# ═════════════════════════════════════════════════════════════════════════════
def generar_test_y_probs(df, xgb_bull, xgb_bear):
    df_clean = df.dropna(subset=FEATURE_COLS + ["target", "regime", "vol_percentile"]).copy()
    n = len(df_clean)
    train_end = int(n * TRAIN_PCT)
    val_end   = int(n * VAL_PCT)
    df_test   = df_clean.iloc[val_end:].copy()
    print(f"[TEST] {df_test.index[0]} -> {df_test.index[-1]} ({len(df_test)} velas)")

    preds, probs, actives = [], [], []
    for _, row in df_test.iterrows():
        regime  = row["regime"]
        vol_pct = row["vol_percentile"]
        x = row[FEATURE_COLS].values.reshape(1, -1)
        model = xgb_bull if regime == "BULL" else xgb_bear
        if vol_pct > VOL_UMBRAL:
            preds.append(np.nan); probs.append(np.nan); actives.append(False); continue
        prob = float(model.predict_proba(x)[0, 1])
        if abs(prob - 0.5) <= 0.05:
            preds.append(np.nan); probs.append(np.nan); actives.append(False)
        else:
            preds.append(1 if prob > 0.5 else 0)
            probs.append(prob)
            actives.append(True)

    df_test["pred"]   = preds
    df_test["prob"]   = probs
    df_test["active"] = actives
    print(f"[TEST] Activas: {sum(actives)} ({sum(actives)/len(df_test):.1%})")
    return df_test


# ═════════════════════════════════════════════════════════════════════════════
# 5. Motores de backtest — copia exacta del notebook
# ═════════════════════════════════════════════════════════════════════════════
def backtest_e6(df, capital_ini):
    capital = capital_ini
    trades = []
    pos = None
    rows = list(df.iterrows())
    for i in range(len(rows) - 1):
        ts, row = rows[i]
        ts_next, rnext = rows[i + 1]
        p_sig = rnext["close"]
        if pos is not None:
            p_ent = pos["entry"]; d = pos["dir"]; size = pos["size"]
            ret = (p_sig - p_ent) / p_ent
            pnl = size * E6_APAL * d * ret - size * (COMISION + SLIPPAGE)
            if pnl < -0.90 * size:
                pnl = -0.90 * size
            capital += pnl
            trades.append({
                "open_ts": pos["open_ts"], "close_ts": ts_next,
                "dir": "LONG" if d == 1 else "SHORT",
                "entry": round(p_ent, 2), "exit": round(p_sig, 2),
                "size": round(size, 4), "prob": round(pos["prob"], 4),
                "pnl": round(pnl, 4), "capital": round(capital, 4),
            })
            pos = None
        prob = row["prob"]
        if not pd.isna(prob) and abs(prob - 0.5) > E6_DELTA:
            d = 1 if prob > 0.5 else -1
            size = capital * E6_PCT
            capital -= size * (COMISION + SLIPPAGE)
            pos = {"open_ts": ts_next, "entry": p_sig, "dir": d, "size": size, "prob": prob}
    return pd.DataFrame(trades) if trades else pd.DataFrame()


def backtest_e11(df, capital_ini):
    capital = capital_ini
    trades = []
    pos = None
    rows = list(df.iterrows())
    for i in range(len(rows) - 1):
        ts, row = rows[i]
        ts_next, rnext = rows[i + 1]
        p_sig = rnext["close"]
        if pos is not None:
            p_ent = pos["entry"]; size = pos["size"]
            ret = (p_sig - p_ent) / p_ent
            pnl = size * E11_APAL * ret - size * (COMISION + SLIPPAGE)
            if pnl < -0.90 * size:
                pnl = -0.90 * size
            capital += pnl
            trades.append({
                "open_ts": pos["open_ts"], "close_ts": ts_next,
                "dir": "LONG", "entry": round(p_ent, 2), "exit": round(p_sig, 2),
                "size": round(size, 4), "pct": round(pos["pct"], 4),
                "prob": round(pos["prob"], 4),
                "pnl": round(pnl, 4), "capital": round(capital, 4),
            })
            pos = None
        prob = row["prob"]
        regime = row["regime"]
        ema_sl = row.get("ema200_slope", 1.0)
        vol_p = row["vol_percentile"]
        if (not pd.isna(prob) and prob - 0.5 > E11_DELTA
                and regime == "BULL" and vol_p <= VOL_UMBRAL
                and ema_sl > 0):
            edge = 2 * (prob - 0.5)
            pct = min(E11_PCT_MAX, E11_BASE_PCT + edge / E11_KELLY_DIV)
            size = capital * pct
            capital -= size * (COMISION + SLIPPAGE)
            pos = {"open_ts": ts_next, "entry": p_sig, "size": size, "pct": pct, "prob": prob}
    return pd.DataFrame(trades) if trades else pd.DataFrame()


# ═════════════════════════════════════════════════════════════════════════════
# 6. Ensamblar output
# ═════════════════════════════════════════════════════════════════════════════
def resumen(name, tr, capital_ini):
    if len(tr) == 0:
        return {"name": name, "n": 0, "pnl": 0, "wr": 0, "avg": 0,
                "best": 0, "worst": 0, "final_capital": capital_ini}
    wins = tr[tr["pnl"] > 0]
    return {
        "name": name,
        "n": int(len(tr)),
        "pnl": float(tr["pnl"].sum()),
        "wr": float(len(wins) / len(tr) * 100),
        "avg": float(tr["pnl"].mean()),
        "best": float(tr["pnl"].max()),
        "worst": float(tr["pnl"].min()),
        "final_capital": float(capital_ini + tr["pnl"].sum()),
    }


def calcular_regime_spans(df_test):
    regime_series = df_test["regime"]
    spans = []
    current_reg = regime_series.iloc[0]
    current_start = regime_series.index[0]
    for ts, reg in regime_series.items():
        if reg != current_reg:
            spans.append({
                "regime": str(current_reg),
                "start": current_start.isoformat(),
                "end": ts.isoformat(),
            })
            current_reg = reg
            current_start = ts
    spans.append({
        "regime": str(current_reg),
        "start": current_start.isoformat(),
        "end": regime_series.index[-1].isoformat(),
    })
    return spans


def trades_to_json(df_tr, strategy):
    out = []
    for _, t in df_tr.iterrows():
        out.append({
            "strategy": strategy,
            "dir": str(t["dir"]),
            "open_ts":  pd.Timestamp(t["open_ts"]).isoformat(),
            "close_ts": pd.Timestamp(t["close_ts"]).isoformat(),
            "entry": float(t["entry"]),
            "exit":  float(t["exit"]),
            "prob":  float(t["prob"]),
            "pnl":   float(t["pnl"]),
        })
    return out


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Cargar XGBoost
    print("[MAIN] Cargando modelos XGBoost...")
    with open(MODELS_DIR / "xgb_bull.pkl", "rb") as f: xgb_bull = pickle.load(f)
    with open(MODELS_DIR / "xgb_bear.pkl", "rb") as f: xgb_bear = pickle.load(f)
    print(f"[MAIN] bull={xgb_bull.n_features_in_}f | bear={xgb_bear.n_features_in_}f")

    # Descargar
    df_btc, df_eth, df_gold, df_dxy, fg = descargar_todo()
    if df_btc.empty:
        print("[MAIN] ERROR: BTC vacio. Abortando.")
        return

    # Features
    df = construir_features(df_btc, df_eth, df_gold, df_dxy, fg)

    # HMM + vol_percentile + EMA slope
    df, df_daily = regimen_hmm(df, df_btc)

    # Test set + probabilidades
    df_test = generar_test_y_probs(df, xgb_bull, xgb_bear)

    # Backtests
    print("[MAIN] Simulando E6...")
    tr_e6 = backtest_e6(df_test, CAPITAL_INI)
    print("[MAIN] Simulando E11...")
    tr_e11 = backtest_e11(df_test, CAPITAL_INI)
    print(f"[MAIN] E6: {len(tr_e6)} trades | E11: {len(tr_e11)} trades")

    # Metricas
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_start": df_test.index[0].isoformat(),
        "test_end":   df_test.index[-1].isoformat(),
        "test_candles": int(len(df_test)),
        "capital_inicial": CAPITAL_INI,
        "e6":  resumen("E6",  tr_e6,  CAPITAL_INI),
        "e11": resumen("E11", tr_e11, CAPITAL_INI),
    }
    print(json.dumps(meta, indent=2, default=str))

    # Regime spans (solo para el periodo test, para pintar bandas)
    regime_spans = calcular_regime_spans(df_test)

    # Candles
    candles = []
    for ts, row in df_test.iterrows():
        candles.append({
            "ts": ts.isoformat(),
            "open":  round(float(row["open"]),  2),
            "high":  round(float(row["high"]),  2),
            "low":   round(float(row["low"]),   2),
            "close": round(float(row["close"]), 2),
            "regime": str(row["regime"]),
        })

    output = {
        "meta": meta,
        "candles": candles,
        "regime_spans": regime_spans,
        "trades_e6":  trades_to_json(tr_e6,  "E6"),
        "trades_e11": trades_to_json(tr_e11, "E11"),
    }

    with open(OUT_FILE, "w") as f:
        json.dump(output, f, separators=(",", ":"))
    sz = os.path.getsize(OUT_FILE) / 1024
    print(f"[MAIN] Guardado {OUT_FILE} ({sz:.1f} KB)")


if __name__ == "__main__":
    main()
