"""
generar_historico.py
====================
Script que genera docs/historical_trades.json con el backtest de E6 y E11
sobre una ventana movil de velas 1h descargadas con yfinance.

Se ejecuta periodicamente en GitHub Actions y publica el resultado para
alimentar la pestana "Estrategias" del dashboard.

Dependencias:
  pip install yfinance pandas numpy hmmlearn xgboost scikit-learn
"""

import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

MODELS_DIR = Path("models")
OUT_FILE   = Path("docs/historical_trades.json")

# Parametros identicos al backtest del TFG
COMISION     = 0.0004
SLIPPAGE     = 0.0005
CAPITAL_INI  = 57.94

E6_DELTA     = 0.30
E6_PCT       = 0.40
E6_APAL      = 5

E11_DELTA        = 0.20
E11_APAL         = 5
E11_BASE_PCT     = 0.20
E11_PCT_MAX      = 0.55
E11_KELLY_DIV    = 3.0

VOL_UMBRAL       = 0.80
VOL_WIN          = 24 * 20  # 20 dias de volatilidad rolling
EMA_SPAN         = 200

# Features XGBoost (26) - mismos nombres que el bot.py
FEATURES_XGB = [
    "log_ret_1h", "log_ret_4h", "log_ret_1d",
    "log_ret_eth", "log_ret_gold", "log_ret_dxy",
    "ema21_diff", "ema50_diff", "sma200_diff",
    "macd", "macd_signal", "macd_hist",
    "rsi_14",
    "roc_6", "roc_12", "roc_24", "mom_12",
    "atr_pct",
    "bb_width", "bb_pct",
    "vol_roll_24h",
    "obv_norm", "volume_zscore",
    "rsi_eth", "corr_btc_eth_24h",
    "fear_greed_norm",
]


# ============================================================================
# Descarga de datos
# ============================================================================
def descargar_yf(ticker, periodo, intervalo):
    print(f"[YF] Descargando {ticker} ({periodo}, {intervalo})...")
    df = yf.download(ticker, period=periodo, interval=intervalo,
                     auto_adjust=False, progress=False)
    if df.empty:
        print(f"[YF] VACIO para {ticker}")
        return df
    # Normalizar columnas: yfinance reciente devuelve MultiIndex a veces
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close",
        "Volume": "volume",
    })
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    print(f"[YF] {ticker}: {len(df)} filas de {df.index.min()} a {df.index.max()}")
    return df


def descargar_todo():
    """Descarga BTC+ETH 1h ultimos 720 dias + Gold+DXY diarios ultimos 2 anos."""
    btc = descargar_yf("BTC-USD", "720d", "1h")
    eth = descargar_yf("ETH-USD", "720d", "1h")
    gold = descargar_yf("GC=F",   "2y",   "1d")
    dxy  = descargar_yf("DX-Y.NYB","2y",   "1d")
    return btc, eth, gold, dxy


# ============================================================================
# Features
# ============================================================================
def alinear_diario_a_1h(df_1h, df_diario, nombre):
    """Expande una serie diaria a frecuencia 1h con forward fill."""
    if df_diario.empty:
        return pd.Series(0.0, index=df_1h.index, name=nombre)
    cerrar = df_diario["close"].copy()
    cerrar.index = cerrar.index.tz_localize("UTC") if cerrar.index.tz is None else cerrar.index
    # Reindexar en 1h y forward fill
    expandido = cerrar.reindex(df_1h.index, method="ffill")
    return expandido


def rsi(serie, periodo=14):
    delta = serie.diff()
    ganancia = delta.where(delta > 0, 0).ewm(alpha=1/periodo, adjust=False).mean()
    perdida = (-delta.where(delta < 0, 0)).ewm(alpha=1/periodo, adjust=False).mean()
    rs = ganancia / perdida.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(serie):
    ema12 = serie.ewm(span=12, adjust=False).mean()
    ema26 = serie.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist


def atr(df, periodo=14):
    h_l = df["high"] - df["low"]
    h_c = (df["high"] - df["close"].shift()).abs()
    l_c = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
    return tr.ewm(alpha=1/periodo, adjust=False).mean()


def bollinger(serie, periodo=20, n_std=2):
    media = serie.rolling(periodo).mean()
    std = serie.rolling(periodo).std()
    upper = media + n_std * std
    lower = media - n_std * std
    width = (upper - lower) / media
    pct = (serie - lower) / (upper - lower)
    return upper, lower, width, pct


def obv(df):
    direccion = np.sign(df["close"].diff()).fillna(0)
    return (direccion * df["volume"]).cumsum()


def construir_features(btc, eth, gold, dxy):
    print("[FEAT] Calculando features...")
    df = btc.copy()

    # Returns
    df["log_ret_1h"] = np.log(df["close"] / df["close"].shift(1))
    df["log_ret_4h"] = np.log(df["close"] / df["close"].shift(4))
    df["log_ret_1d"] = np.log(df["close"] / df["close"].shift(24))

    # EMAs / SMAs
    df["ema21"]  = df["close"].ewm(span=21).mean()
    df["ema50"]  = df["close"].ewm(span=50).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    df["ema21_diff"]  = (df["close"] - df["ema21"]) / df["ema21"]
    df["ema50_diff"]  = (df["close"] - df["ema50"]) / df["ema50"]
    df["sma200_diff"] = (df["close"] - df["sma200"]) / df["sma200"]

    # MACD
    m, s, h = macd(df["close"])
    df["macd"] = m
    df["macd_signal"] = s
    df["macd_hist"] = h

    # RSI
    df["rsi_14"] = rsi(df["close"], 14)

    # ROC / momentum
    df["roc_6"]  = df["close"].pct_change(6)
    df["roc_12"] = df["close"].pct_change(12)
    df["roc_24"] = df["close"].pct_change(24)
    df["mom_12"] = df["close"] - df["close"].shift(12)

    # ATR
    atr_val = atr(df, 14)
    df["atr_pct"] = atr_val / df["close"]

    # Bollinger
    _, _, bbw, bbp = bollinger(df["close"], 20, 2)
    df["bb_width"] = bbw
    df["bb_pct"] = bbp

    # Volatilidad rolling y volume
    df["vol_roll_24h"] = df["log_ret_1h"].rolling(24).std()
    obv_raw = obv(df)
    df["obv_norm"] = (obv_raw - obv_raw.rolling(100).mean()) / obv_raw.rolling(100).std()
    df["volume_zscore"] = (df["volume"] - df["volume"].rolling(100).mean()) / df["volume"].rolling(100).std()

    # ETH
    if not eth.empty:
        eth_1h_close = eth["close"].reindex(df.index, method="ffill")
        df["log_ret_eth"] = np.log(eth_1h_close / eth_1h_close.shift(1))
        df["rsi_eth"] = rsi(eth_1h_close, 14)
        df["corr_btc_eth_24h"] = df["log_ret_1h"].rolling(24).corr(df["log_ret_eth"])
    else:
        df["log_ret_eth"] = 0.0
        df["rsi_eth"] = 50.0
        df["corr_btc_eth_24h"] = 0.0

    # Gold / DXY diarios
    gold_close = alinear_diario_a_1h(df, gold, "gold_close")
    dxy_close  = alinear_diario_a_1h(df, dxy,  "dxy_close")
    df["log_ret_gold"] = np.log(gold_close / gold_close.shift(24))
    df["log_ret_dxy"]  = np.log(dxy_close  / dxy_close.shift(24))

    # Fear & Greed (no disponible en backtest historico de forma facil):
    # dejamos 0.5 como valor neutral. Si luego se quiere integrar el historico
    # de alternative.me se puede enganchar aqui.
    df["fear_greed_norm"] = 0.5

    # Limpieza
    df = df.dropna(subset=FEATURES_XGB + ["close"])
    print(f"[FEAT] Filas utiles con todas las features: {len(df)}")
    return df


# ============================================================================
# HMM: predecir regimen sobre todas las filas
# ============================================================================
def predecir_regimen(df, hmm_bundle):
    print("[HMM] Prediciendo regimenes...")
    hmm = hmm_bundle["hmm"]
    scaler = hmm_bundle["scaler"]
    feats_hmm = hmm_bundle["features"]
    state_to_regime = hmm_bundle["state_to_regime"]

    # realized_vol_24h ya esta como vol_roll_24h
    df_hmm = df.copy()
    df_hmm["abs_log_ret_1h"] = df_hmm["log_ret_1h"].abs()
    df_hmm["realized_vol_24h"] = df_hmm["vol_roll_24h"]

    X = df_hmm[feats_hmm].values
    X_scaled = scaler.transform(X)
    states = hmm.predict(X_scaled)

    df["regime"] = [state_to_regime[int(s)] for s in states]

    # Slope EMA200 (pendiente 5 dias)
    df["ema200_slope"] = df["sma200"].diff(24 * 5)

    # Volatilidad percentil
    df["vol_percentile"] = df["vol_roll_24h"].rank(pct=True)
    print(f"[HMM] Distribucion de regimenes:\n{df['regime'].value_counts(normalize=True)}")
    return df


# ============================================================================
# Predicciones XGBoost segun regimen
# ============================================================================
def predecir_xgb(df, xgb_bull, xgb_bear):
    print("[XGB] Prediciendo probabilidades...")
    X = df[FEATURES_XGB].values

    prob_bull = xgb_bull.predict_proba(X)[:, 1]
    prob_bear = xgb_bear.predict_proba(X)[:, 1]

    df["prob_bull"] = prob_bull
    df["prob_bear"] = prob_bear

    # prob segun regimen del momento
    df["prob_xgb"] = np.where(
        df["regime"] == "BULL",
        prob_bull,
        prob_bear,
    )
    return df


# ============================================================================
# Simulacion de trades
# ============================================================================
def kelly_fraccion(prob):
    """Kelly simple para prob binaria con ganancia~perdida."""
    if prob <= 0.5:
        return 0.0
    edge = 2 * prob - 1.0
    return edge / E11_KELLY_DIV


def ejecutar_trade(idx_entrada, df, capital, side, apal, pct):
    """Abre en close[i], cierra en close[i+1] (como en el backtest)."""
    if idx_entrada >= len(df) - 1:
        return None
    row_in = df.iloc[idx_entrada]
    row_out = df.iloc[idx_entrada + 1]

    entry_px = row_in["close"]
    exit_px  = row_out["close"]

    if side == "LONG":
        ret = (exit_px - entry_px) / entry_px
    else:
        ret = (entry_px - exit_px) / entry_px

    ret_neto = ret * apal - 2 * COMISION * apal - SLIPPAGE * apal
    ret_neto = max(ret_neto, -0.90)  # cap -90%
    pnl_pct = ret_neto * pct * 100
    pnl_usdt = capital * pct * ret_neto

    return {
        "strategy": None,  # asignar fuera
        "side": side,
        "open_time": df.index[idx_entrada].isoformat(),
        "close_time": df.index[idx_entrada + 1].isoformat(),
        "entry_px": round(float(entry_px), 2),
        "exit_px": round(float(exit_px), 2),
        "pnl_pct": round(float(pnl_pct), 4),
        "pnl_usdt": round(float(pnl_usdt), 4),
        "size_pct": round(float(pct), 4),
        "apal": apal,
    }


def simular_e11(df):
    print("[SIM] Simulando E11...")
    trades = []
    capital = CAPITAL_INI
    i = 0
    while i < len(df) - 1:
        row = df.iloc[i]
        regime = row["regime"]
        vol_pct = row["vol_percentile"]
        ema_slope = row["ema200_slope"]
        prob = row["prob_xgb"]

        # Condicion E11: BULL + Vol OK + EMA200 up + prob >= 0.70
        if (regime == "BULL"
                and vol_pct <= VOL_UMBRAL
                and ema_slope is not None and ema_slope > 0
                and prob >= 0.70):
            kelly = kelly_fraccion(prob)
            pct = min(max(E11_BASE_PCT + kelly, E11_BASE_PCT), E11_PCT_MAX)
            trade = ejecutar_trade(i, df, capital, "LONG", E11_APAL, pct)
            if trade is not None:
                trade["strategy"] = "E11"
                trade["prob"] = round(float(prob), 4)
                trades.append(trade)
                capital += trade["pnl_usdt"]
                i += 2  # saltar una vela (trade ocupa i e i+1)
                continue
        i += 1

    print(f"[SIM] E11: {len(trades)} trades | capital {CAPITAL_INI:.2f} -> {capital:.2f}")
    return trades


def simular_e6(df):
    print("[SIM] Simulando E6...")
    trades = []
    capital = CAPITAL_INI
    i = 0
    while i < len(df) - 1:
        row = df.iloc[i]
        regime = row["regime"]
        vol_pct = row["vol_percentile"]
        prob = row["prob_xgb"]

        # E6: Vol OK + (prob >= 0.80 LONG) o (prob <= 0.20 SHORT y regimen != BULL)
        if vol_pct <= VOL_UMBRAL:
            side = None
            if prob >= 0.80:
                side = "LONG"
            elif prob <= 0.20 and regime != "BULL":
                side = "SHORT"
            if side:
                trade = ejecutar_trade(i, df, capital, side, E6_APAL, E6_PCT)
                if trade is not None:
                    trade["strategy"] = "E6"
                    trade["prob"] = round(float(prob), 4)
                    trades.append(trade)
                    capital += trade["pnl_usdt"]
                    i += 2
                    continue
        i += 1

    print(f"[SIM] E6: {len(trades)} trades | capital {CAPITAL_INI:.2f} -> {capital:.2f}")
    return trades


def regime_spans(df):
    """Convierte la serie de regimen en spans continuos para las bandas."""
    spans = []
    current = None
    start = None
    for ts, row in df.iterrows():
        reg = row["regime"]
        if reg != current:
            if current is not None:
                spans.append({"start": start.isoformat(), "end": ts.isoformat(), "regime": current})
            current = reg
            start = ts
    if current is not None:
        spans.append({"start": start.isoformat(), "end": df.index[-1].isoformat(), "regime": current})
    return spans


# ============================================================================
# Main
# ============================================================================
def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Cargar modelos
    print("[MAIN] Cargando modelos...")
    with open(MODELS_DIR / "hmm_model.pkl", "rb") as f:
        hmm_bundle = pickle.load(f)
    with open(MODELS_DIR / "xgb_bull.pkl", "rb") as f:
        xgb_bull = pickle.load(f)
    with open(MODELS_DIR / "xgb_bear.pkl", "rb") as f:
        xgb_bear = pickle.load(f)
    print(f"[MAIN] HMM entrenado en: {hmm_bundle.get('trained_on', {})}")

    # Descargar datos
    btc, eth, gold, dxy = descargar_todo()
    if btc.empty:
        print("[MAIN] Error: BTC 1h vacio. Abortando.")
        sys.exit(1)

    # Features
    df = construir_features(btc, eth, gold, dxy)

    # Regimen HMM
    df = predecir_regimen(df, hmm_bundle)

    # XGBoost
    df = predecir_xgb(df, xgb_bull, xgb_bear)

    # Cortar warmup: eliminar primeras 200 filas por SMA200
    df = df.iloc[200:].dropna(subset=["ema200_slope", "regime"])
    print(f"[MAIN] Filas tras warmup: {len(df)}")

    # Simular
    trades_e11 = simular_e11(df)
    trades_e6 = simular_e6(df)

    # Spans de regimen
    spans = regime_spans(df)
    print(f"[MAIN] Spans de regimen: {len(spans)}")

    # Candles para el grafico (downsample a cada hora ya)
    candles = []
    for ts, row in df.iterrows():
        candles.append({
            "t": ts.isoformat(),
            "o": round(float(row["open"]), 2),
            "h": round(float(row["high"]), 2),
            "l": round(float(row["low"]), 2),
            "c": round(float(row["close"]), 2),
        })

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": {
            "date_range": f"{df.index.min().date()} / {df.index.max().date()}",
            "rows": len(df),
            "source": "yfinance 720d 1h",
            "e11_trades": len(trades_e11),
            "e6_trades": len(trades_e6),
        },
        "candles": candles,
        "regime_spans": spans,
        "trades_e11": trades_e11,
        "trades_e6": trades_e6,
    }

    with open(OUT_FILE, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"[MAIN] Escrito {OUT_FILE}")
    print(f"[MAIN] Tamano: {os.path.getsize(OUT_FILE) / 1024:.1f} KB")
    print(f"[MAIN] E11: {len(trades_e11)} trades | E6: {len(trades_e6)} trades")


if __name__ == "__main__":
    main()
