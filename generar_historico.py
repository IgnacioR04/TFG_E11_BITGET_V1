"""
generar_historico.py
====================
Replica operativa del notebook comparativa_e6_e11.ipynb.
Entrena el HMM sobre datos diarios, aplica el pipeline y simula la cascada E11 -> E6 sobre el test set.

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
OUT_FILE = Path("docs/historical_trades.json")

# ── Parametros identicos a comparativa_e6_e11.ipynb ──────────────────────────
COMISION = 0.0004
SLIPPAGE = 0.0005
CAPITAL_INI = 57.94
TRAIN_PCT = 0.70
VAL_PCT = 0.85

# E6
E6_DELTA = 0.30
E6_APAL = 5
E6_PCT = 0.40

# E11
E11_DELTA = 0.20
E11_APAL = 5
E11_BASE_PCT = 0.20
E11_PCT_MAX = 0.55
E11_KELLY_DIV = 3.0
ALLOW_SHORTS = False
REQUIRE_MACRO = True

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
# 1. Descarga de datos via yfinance
# ═════════════════════════════════════════════════════════════════════════════
def descargar_yf(ticker, periodo, intervalo):
    print(f"[YF] Descargando {ticker} ({periodo}, {intervalo})...")
    df = yf.download(
        ticker,
        period=periodo,
        interval=intervalo,
        auto_adjust=False,
        progress=False,
    )
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


def descargar_yf_1h_largo(ticker, start="2019-01-01", chunk_days=680):
    print(f"[YF] Descargando largo {ticker} 1h desde {start}...")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp.utcnow().tz_localize(None)

    partes = []
    cur = start_ts

    while cur < end_ts:
        nxt = min(cur + pd.Timedelta(days=chunk_days), end_ts)
        print(f"[YF]  tramo {ticker} {cur.date()} -> {nxt.date()}")

        df = yf.download(
            ticker,
            start=cur.strftime("%Y-%m-%d"),
            end=nxt.strftime("%Y-%m-%d"),
            interval="1h",
            auto_adjust=False,
            progress=False,
        )

        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            df.index = pd.to_datetime(df.index)

            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)

            df.index.name = "datetime"
            partes.append(df)

        cur = nxt - pd.Timedelta(days=2)

    if not partes:
        return pd.DataFrame()

    out = pd.concat(partes).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def descargar_todo():
    # Descarga larga por tramos para 1h
    df_btc = descargar_yf_1h_largo("BTC-USD", start="2019-01-01")
    df_eth = descargar_yf_1h_largo("ETH-USD", start="2019-01-01")

    # Macros con mucha mas historia para no recortar el dataset
    df_gold = descargar_yf("GC=F", "10y", "1d")
    df_dxy = descargar_yf("DX-Y.NYB", "10y", "1d")

    df_btc = df_btc[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    df_eth = df_eth[["close"]].rename(columns={"close": "eth"})
    df_gold = df_gold[["close"]].rename(columns={"close": "gold"})
    df_dxy = df_dxy[["close"]].rename(columns={"close": "dxy"})

    # Fear & Greed completo
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=0&format=json", timeout=10)
        dat = r.json()["data"]
        fg = pd.DataFrame(dat)[["timestamp", "value"]]
        fg["timestamp"] = pd.to_datetime(fg["timestamp"].astype(int), unit="s")
        fg = fg.set_index("timestamp").sort_index()
        fg["value"] = fg["value"].astype(float)
        fg.index.name = "datetime"
    except Exception as e:
        print(f"[FNG] Error {e}. Usando valor neutro 50.")
        fg = pd.DataFrame({"value": [50]}, index=[df_btc.index[0]])

    return df_btc, df_eth, df_gold, df_dxy, fg


# ═════════════════════════════════════════════════════════════════════════════
# 2. Feature engineering
# ═════════════════════════════════════════════════════════════════════════════
def construir_features(df_btc, df_eth, df_gold, df_dxy, fg):
    print("[FEAT] Calculando features...")
    df = df_btc[["open", "high", "low", "close", "volume"]].copy()

    for asset in [df_eth, df_gold, df_dxy]:
        df = df.join(asset.reindex(df.index, method="ffill"), how="left")

    df["fear_greed"] = fg.reindex(df.index, method="ffill")["value"]
    df["log_ret_1h"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna(subset=["close", "log_ret_1h"])

    df["log_ret_4h"] = df["log_ret_1h"].rolling(4).sum()
    df["log_ret_1d"] = df["log_ret_1h"].rolling(24).sum()
    df["log_ret_eth"] = np.log(df["eth"] / df["eth"].shift(1))
    df["log_ret_gold"] = np.log(df["gold"] / df["gold"].shift(1)).ffill()
    df["log_ret_dxy"] = np.log(df["dxy"] / df["dxy"].shift(1)).ffill()

    close = df["close"]
    ema21 = ta.ema(close, length=21)
    ema50 = ta.ema(close, length=50)
    sma200 = ta.sma(close, length=200)
    df["ema21_diff"] = (close - ema21) / close
    df["ema50_diff"] = (close - ema50) / close
    df["sma200_diff"] = (close - sma200) / close

    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    df["macd"] = macd_df["MACD_12_26_9"] / close
    df["macd_signal"] = macd_df["MACDs_12_26_9"] / close
    df["macd_hist"] = macd_df["MACDh_12_26_9"] / close

    df["rsi_14"] = ta.rsi(close, length=14)
    df["roc_6"] = ta.roc(close, length=6)
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
    df["bb_pct"] = (close - bb[bb_l]) / (bb[bb_u] - bb[bb_l])

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
    print(f"[FEAT] Shape final {df.shape}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 3. HMM sobre datos diarios
# ═════════════════════════════════════════════════════════════════════════════
def regimen_hmm(df, df_btc):
    print("[HMM] Resampleando a diario y entrenando...")

    df_btc_1d = df_btc.resample("1D").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["close"])

    df_daily = df_btc_1d.copy()
    df_daily["log_ret"] = np.log(df_daily["close"] / df_daily["close"].shift(1))
    df_daily["vol_roll_20d"] = df_daily["log_ret"].rolling(20).std()
    vm_d = df_daily["volume"].rolling(20).mean()
    vs_d = df_daily["volume"].rolling(20).std()
    df_daily["volume_zscore"] = (df_daily["volume"] - vm_d) / (vs_d + 1e-9)
    df_daily = df_daily.dropna()

    X_hmm = df_daily[["log_ret", "vol_roll_20d", "volume_zscore"]].copy()
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_hmm)

    best_m = None
    best_s = -np.inf
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
            print(f"[HMM] seed {i} fallo {e}")

    print(f"[HMM] Mejor log-likelihood {best_s:.2f}")

    states = best_m.predict(X_sc)
    df_daily["state_raw"] = states

    mean_ret = df_daily.groupby("state_raw")["log_ret"].mean()
    sorted_s = mean_ret.sort_values().index.tolist()
    state_map = {sorted_s[0]: "BEAR", sorted_s[1]: "SIDEWAYS", sorted_s[2]: "BULL"}
    df_daily["regime"] = df_daily["state_raw"].map(state_map)

    print(f"[HMM] Distribucion {df_daily['regime'].value_counts().to_dict()}")

    reg_d = df_daily[["regime"]].copy()
    reg_d.index = pd.to_datetime(reg_d.index).normalize()
    rm = reg_d.reindex(df.index.normalize(), method="ffill")
    rm.index = df.index
    df["regime"] = rm["regime"].values

    df_daily["vol_percentile"] = df_daily["vol_roll_20d"].expanding().rank(pct=True)
    vp_d = df_daily[["vol_percentile"]].copy()
    vp_d.index = pd.to_datetime(vp_d.index).normalize()
    vp_m = vp_d.reindex(df.index.normalize(), method="ffill")
    vp_m.index = df.index
    df["vol_percentile"] = vp_m["vol_percentile"].values

    ema200 = df_daily["close"].ewm(span=200, adjust=False).mean()
    ema200_slope = ema200 - ema200.shift(5)
    es_d = pd.DataFrame({"ema_slope": ema200_slope})
    es_d.index = pd.to_datetime(es_d.index).normalize()
    es_m = es_d.reindex(df.index.normalize(), method="ffill")
    es_m.index = df.index
    df["ema200_slope"] = es_m["ema_slope"].values

    return df, df_daily


# ═════════════════════════════════════════════════════════════════════════════
# 4. Split y probabilidades
# ═════════════════════════════════════════════════════════════════════════════
def generar_test_y_probs(df, xgb_bull, xgb_bear):
    df_clean = df.dropna(subset=FEATURE_COLS + ["target", "regime", "vol_percentile"]).copy()
    n = len(df_clean)
    train_end = int(n * TRAIN_PCT)
    val_end = int(n * VAL_PCT)
    df_test = df_clean.iloc[val_end:].copy()
    print(f"[TEST] {df_test.index[0]} -> {df_test.index[-1]} ({len(df_test)} velas)")

    preds = []
    probs = []
    actives = []

    for _, row in df_test.iterrows():
        regime = row["regime"]
        vol_pct = row["vol_percentile"]
        x = row[FEATURE_COLS].values.reshape(1, -1)
        model = xgb_bull if regime == "BULL" else xgb_bear

        if vol_pct > VOL_UMBRAL:
            preds.append(np.nan)
            probs.append(np.nan)
            actives.append(False)
            continue

        prob = float(model.predict_proba(x)[0, 1])

        if abs(prob - 0.5) <= 0.05:
            preds.append(np.nan)
            probs.append(np.nan)
            actives.append(False)
        else:
            preds.append(1 if prob > 0.5 else 0)
            probs.append(prob)
            actives.append(True)

    df_test["pred"] = preds
    df_test["prob"] = probs
    df_test["active"] = actives
    print(f"[TEST] Activas {sum(actives)} ({sum(actives) / len(df_test):.1%})")

    probs_validas = pd.Series([p for p in probs if pd.notna(p)])
    if len(probs_validas) > 0:
        print("[TEST] Prob cuantiles")
        print(probs_validas.quantile([0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]))

    print("[TEST] Regimen test")
    print(df_test["regime"].value_counts(dropna=False).to_dict())

    return df_test


# ═════════════════════════════════════════════════════════════════════════════
# 5. Motores de backtest
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
            p_ent = pos["entry"]
            d = pos["dir"]
            size = pos["size"]
            ret = (p_sig - p_ent) / p_ent
            pnl = size * E6_APAL * d * ret - size * (COMISION + SLIPPAGE)
            if pnl < -0.90 * size:
                pnl = -0.90 * size
            capital += pnl
            trades.append({
                "open_ts": pos["open_ts"],
                "close_ts": ts_next,
                "dir": "LONG" if d == 1 else "SHORT",
                "entry": round(p_ent, 2),
                "exit": round(p_sig, 2),
                "size": round(size, 4),
                "prob": round(pos["prob"], 4),
                "pnl": round(pnl, 4),
                "capital": round(capital, 4),
            })
            pos = None

        prob = row["prob"]
        if row.get("active", False) and not pd.isna(prob) and abs(prob - 0.5) > E6_DELTA:
            d = 1 if prob > 0.5 else -1
            size = capital * E6_PCT
            capital -= size * (COMISION + SLIPPAGE)
            pos = {
                "open_ts": ts_next,
                "entry": p_sig,
                "dir": d,
                "size": size,
                "prob": prob,
            }

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
            p_ent = pos["entry"]
            d = pos["dir"]
            size = pos["size"]
            ret = (p_sig - p_ent) / p_ent
            pnl = size * E11_APAL * d * ret - size * (COMISION + SLIPPAGE)
            if pnl < -0.90 * size:
                pnl = -0.90 * size
            capital += pnl
            trades.append({
                "open_ts": pos["open_ts"],
                "close_ts": ts_next,
                "dir": "LONG" if d == 1 else "SHORT",
                "entry": round(p_ent, 2),
                "exit": round(p_sig, 2),
                "size": round(size, 4),
                "pct": round(pos["pct"], 4),
                "prob": round(pos["prob"], 4),
                "pnl": round(pnl, 4),
                "capital": round(capital, 4),
            })
            pos = None

        if row.get("active", False):
            prob = row["prob"]

            if pd.isna(prob) or abs(prob - 0.5) <= E11_DELTA:
                continue

            if str(row.get("regime", "")) != "BULL":
                continue

            d = 1 if prob > 0.5 else -1

            if d == -1 and not ALLOW_SHORTS:
                continue

            if d == 1 and row.get("ema200_slope", 1.0) <= 0:
                continue

            edge = 2 * abs(prob - 0.5)
            pct = min(E11_PCT_MAX, E11_BASE_PCT + edge / E11_KELLY_DIV)
            size = capital * pct
            capital -= size * (COMISION + SLIPPAGE)
            pos = {
                "open_ts": ts_next,
                "entry": p_sig,
                "dir": d,
                "size": size,
                "pct": pct,
                "prob": prob,
            }

    if len(trades) == 0:
        tmp = df.copy()
        tmp = tmp[tmp["active"] == True].copy()
        if len(tmp) > 0:
            tmp["abs_edge"] = (tmp["prob"] - 0.5).abs()
            n_delta = int((tmp["abs_edge"] > E11_DELTA).sum())
            n_long = int((tmp["prob"] > 0.5).sum())
            n_macro = int(((tmp["prob"] > 0.5) & (tmp["ema200_slope"] > 0)).sum())
            print(f"[E11 DEBUG] activas={len(tmp)}  pasan_delta={n_delta}  longs={n_long}  longs_macro_ok={n_macro}")

    return pd.DataFrame(trades) if trades else pd.DataFrame()


# ═════════════════════════════════════════════════════════════════════════════
# 6. Ensamblar output
# ═════════════════════════════════════════════════════════════════════════════
def _equity_curve_from_trades(df_tr, capital_ini):
    if len(df_tr) == 0:
        return [capital_ini]
    vals = [capital_ini]
    vals.extend([float(x) for x in df_tr["capital"].tolist()])
    return vals


def _max_drawdown_pct(equity_values):
    arr = np.asarray(equity_values, dtype=float)
    if arr.size == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / np.where(peak == 0, 1.0, peak)
    return float(dd.min() * 100.0)


def resumen(name, tr, capital_ini):
    if len(tr) == 0:
        return {
            "name": name,
            "trades": 0,
            "win_rate": 0.0,
            "total_pnl_usdt": 0.0,
            "avg_pnl_usdt": 0.0,
            "best_pnl_usdt": 0.0,
            "worst_pnl_usdt": 0.0,
            "final_capital": float(capital_ini),
            "ret_total_pct": 0.0,
            "avg_return_on_margin_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "avg_size_pct": 0.0,
            "size_pct_min": 0.0,
            "size_pct_max": 0.0,
        }

    wins = tr[tr["pnl"] > 0]
    total_pnl = float(tr["pnl"].sum())
    final_cap = float(tr["capital"].iloc[-1])
    ret_total_pct = (final_cap / capital_ini - 1.0) * 100.0
    long_trades = int((tr["dir"] == "LONG").sum()) if "dir" in tr.columns else 0
    short_trades = int((tr["dir"] == "SHORT").sum()) if "dir" in tr.columns else 0

    size_col = "pct" if "pct" in tr.columns else None
    if size_col is None and "size" in tr.columns:
        size_pct_series = tr["size"] / tr["capital"].shift(1).fillna(capital_ini)
    elif size_col is not None:
        size_pct_series = tr[size_col]
    else:
        size_pct_series = pd.Series([0.0] * len(tr), index=tr.index)

    margin_ret_pct = (tr["pnl"] / tr["size"].replace(0, np.nan) * 100.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    eq_vals = _equity_curve_from_trades(tr, capital_ini)

    return {
        "name": name,
        "trades": int(len(tr)),
        "win_rate": float(len(wins) / len(tr) * 100.0),
        "total_pnl_usdt": total_pnl,
        "avg_pnl_usdt": float(tr["pnl"].mean()),
        "best_pnl_usdt": float(tr["pnl"].max()),
        "worst_pnl_usdt": float(tr["pnl"].min()),
        "final_capital": final_cap,
        "ret_total_pct": float(ret_total_pct),
        "avg_return_on_margin_pct": float(margin_ret_pct.mean()),
        "max_drawdown_pct": _max_drawdown_pct(eq_vals),
        "long_trades": long_trades,
        "short_trades": short_trades,
        "avg_size_pct": float(size_pct_series.mean() * 100.0),
        "size_pct_min": float(size_pct_series.min() * 100.0),
        "size_pct_max": float(size_pct_series.max() * 100.0),
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
        side = str(t["dir"])
        size_pct = float(t["pct"] * 100.0) if "pct" in t and not pd.isna(t.get("pct", np.nan)) else None
        pnl_usdt = float(t["pnl"])
        margin = float(t["size"]) if float(t.get("size", 0.0)) != 0 else np.nan
        pnl_pct = float(pnl_usdt / margin * 100.0) if pd.notna(margin) else 0.0

        row = {
            "strategy": strategy,
            "side": side,
            "dir": side,
            "open_time": pd.Timestamp(t["open_ts"]).isoformat(),
            "close_time": pd.Timestamp(t["close_ts"]).isoformat(),
            "open_ts": pd.Timestamp(t["open_ts"]).isoformat(),
            "close_ts": pd.Timestamp(t["close_ts"]).isoformat(),
            "entry_px": float(t["entry"]),
            "exit_px": float(t["exit"]),
            "entry": float(t["entry"]),
            "exit": float(t["exit"]),
            "prob": float(t["prob"]),
            "pnl_usdt": pnl_usdt,
            "pnl": pnl_usdt,
            "pnl_pct": pnl_pct,
            "size_usdt": float(t["size"]),
            "size": float(t["size"]),
            "size_pct": size_pct,
            "pct": float(t["pct"]) if "pct" in t and not pd.isna(t.get("pct", np.nan)) else None,
            "apal": E11_APAL if strategy == "E11" else E6_APAL,
        }
        out.append(row)
    return out


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("[MAIN] Cargando modelos XGBoost...")
    with open(MODELS_DIR / "xgb_bull.pkl", "rb") as f:
        xgb_bull = pickle.load(f)
    with open(MODELS_DIR / "xgb_bear.pkl", "rb") as f:
        xgb_bear = pickle.load(f)
    print(f"[MAIN] bull={xgb_bull.n_features_in_}f | bear={xgb_bear.n_features_in_}f")

    df_btc, df_eth, df_gold, df_dxy, fg = descargar_todo()
    if df_btc.empty:
        print("[MAIN] ERROR BTC vacio. Abortando.")
        return

    df = construir_features(df_btc, df_eth, df_gold, df_dxy, fg)
    df, df_daily = regimen_hmm(df, df_btc)
    df_test = generar_test_y_probs(df, xgb_bull, xgb_bear)

    print("[MAIN] Simulando E6...")
    tr_e6 = backtest_e6(df_test, CAPITAL_INI)
    print("[MAIN] Simulando E11...")
    tr_e11 = backtest_e11(df_test, CAPITAL_INI)
    print(f"[MAIN] E6={len(tr_e6)} trades | E11={len(tr_e11)} trades")

    sum_e6 = resumen("E6", tr_e6, CAPITAL_INI)
    sum_e11 = resumen("E11", tr_e11, CAPITAL_INI)
    test_start = df_test.index[0].isoformat()
    test_end = df_test.index[-1].isoformat()

    active_df = df_test[df_test["active"] == True].copy()
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "logic_version": "E11_BULL_GATE_v1",
        "date_range": f"{str(df_test.index[0].date())} / {str(df_test.index[-1].date())}",
        "test_start": test_start,
        "test_end": test_end,
        "test_candles": int(len(df_test)),
        "rows": int(len(df_test)),
        "source": "yfinance largo por tramos 1h",
        "capital_inicial": CAPITAL_INI,
        "e6": sum_e6,
        "e11": sum_e11,
        "e6_trades": int(sum_e6["trades"]),
        "e11_trades": int(sum_e11["trades"]),
        "e11_rejections": {
            "active_rows": int(len(active_df)),
            "dead_zone_e11_or_nan": int(((df_test["active"] == True) & (df_test["prob"].isna() | ((df_test["prob"] - 0.5).abs() <= E11_DELTA))).sum()),
            "not_bull": int(((active_df["prob"] - 0.5).abs() > E11_DELTA).mul(active_df["regime"] != "BULL").sum()),
            "ema_not_ok": int((((active_df["prob"] - 0.5).abs() > E11_DELTA) & (active_df["regime"] == "BULL") & (active_df["prob"] > 0.5) & (active_df["ema200_slope"] <= 0)).sum()),
            "shorts_blocked": int((((active_df["prob"] - 0.5).abs() > E11_DELTA) & (active_df["regime"] == "BULL") & (active_df["prob"] < 0.5)).sum()),
        },
    }
    print(json.dumps(meta, indent=2, default=str))

    regime_spans = calcular_regime_spans(df_test)

    candles = []
    for ts, row in df_test.iterrows():
        candles.append({
            "t": ts.isoformat(),
            "ts": ts.isoformat(),
            "o": round(float(row["open"]), 2),
            "h": round(float(row["high"]), 2),
            "l": round(float(row["low"]), 2),
            "c": round(float(row["close"]), 2),
            "open": round(float(row["open"]), 2),
            "high": round(float(row["high"]), 2),
            "low": round(float(row["low"]), 2),
            "close": round(float(row["close"]), 2),
            "regime": str(row["regime"]),
        })

    output = {
        "generated_at": meta["generated_at"],
        "meta": meta,
        "candles": candles,
        "regime_spans": regime_spans,
        "trades_e6": trades_to_json(tr_e6, "E6"),
        "trades_e11": trades_to_json(tr_e11, "E11"),
    }

    with open(OUT_FILE, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    sz = os.path.getsize(OUT_FILE) / 1024
    print(f"[MAIN] Guardado {OUT_FILE} ({sz:.1f} KB)")


if __name__ == "__main__":
    main()
