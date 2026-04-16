"""
bot.py — E11 XGBoost Bot
=========================
Pipeline: descarga datos 1h → calcula 26 features → filtros HMM/GARCH/EMA200 → XGBoost → Kelly sizing → ejecuta en Bitget
Se ejecuta cada minuto via GitHub Actions. Solo opera LONG cuando todos los filtros pasan.
La posicion se abre y cierra en la misma hora (holding = 1 ciclo de senal).
"""

import json
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ── Deteccion de modo live ────────────────────────────────────────────────────
LIVE_MODE = bool(os.environ.get("BITGET_API_KEY"))
if LIVE_MODE:
    try:
        from bitget_api import client_from_env
        print("[E11] Credenciales Bitget detectadas — modo LIVE activo")
    except ImportError:
        LIVE_MODE = False
        print("[E11] bitget_api.py no encontrado — solo paper")

# ── Parametros estrategia E11 ─────────────────────────────────────────────────
DELTA           = 0.20    # dead zone XGBoost
LEVERAGE        = 5       # apalancamiento fijo
BASE_PCT        = 0.20    # sizing minimo (20% del capital)
PCT_MAX         = 0.55    # sizing maximo (55% del capital)
KELLY_DIV       = 3.0     # divisor Kelly fraccional
GARCH_VOL_UMBRAL= 0.80    # percentil de corte de volatilidad
EMA_SPAN        = 200     # periodo EMA tendencia diaria
EMA_SLOPE_DAYS  = 5       # ventana pendiente EMA200
ALLOW_SHORTS    = False   # shorts desactivados
COMISION        = 0.0004  # 0.04% por lado

PARAMS_BULL = {
    'n_estimators': 203,
    'max_depth': 3,
    'learning_rate': 0.012303727625885412,
    'subsample': 0.9989516455010893,
    'colsample_bytree': 0.4156885828019996,
    'min_child_weight': 20,
    'scale_pos_weight': 1.838089273033983
}

PARAMS_BEAR = {
    'n_estimators': 201,
    'max_depth': 7,
    'learning_rate': 0.016121532375941407,
    'subsample': 0.5013167040343544,
    'colsample_bytree': 0.5345656836156467,
    'min_child_weight': 12,
    'scale_pos_weight': 1.2907125770828312
}

FEATURE_COLS = [
    'log_ret_1h', 'log_ret_4h', 'log_ret_1d',
    'log_ret_eth', 'log_ret_gold', 'log_ret_dxy',
    'ema21_diff', 'ema50_diff', 'sma200_diff',
    'macd', 'macd_signal', 'macd_hist',
    'rsi_14',
    'roc_6', 'roc_12', 'roc_24',
    'mom_12',
    'atr_pct', 'bb_width', 'bb_pct', 'vol_roll_24h',
    'obv_norm', 'volume_zscore',
    'rsi_eth', 'corr_btc_eth_24h',
    'fear_greed_norm'
]

STATE_FILE = Path("state.json")
DATA_FILE  = Path("docs/data.json")

# ── Estado inicial ────────────────────────────────────────────────────────────
def get_initial_state(equity_from_exchange=0.0):
    return {
        "runs": 0,
        "last_run": None,
        "paused": False,
        "pause_reason": None,
        "consecutive_errors": 0,
        "consecutive_losses": 0,
        "initial_equity": round(equity_from_exchange, 4),
        "position": {
            "open": False,
            "side": None,
            "entry_price": None,
            "size_btc": None,
            "size_usdt": None,
            "pct_used": None,
            "leverage": None,
            "open_time": None,
            "order_id": None
        },
        "last_signal": None,
        "last_filters": {},
        "trades": [],
        "equity_history": [],
        "live_balance": {"available": 0, "equity": 0, "unrealized_pl": 0}
    }

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    # Si no existe state.json, leer equity de Bitget para inicializar
    print("[E11] state.json no existe, inicializando...")
    initial_eq = 0.0
    if LIVE_MODE:
        try:
            client = client_from_env()
            bal = client.get_balance()
            initial_eq = bal["equity"]
            print(f"[E11] Equity inicial leido de Bitget: {initial_eq:.4f} USDT")
        except Exception as e:
            print(f"[E11] No se pudo leer balance inicial: {e}")
    return get_initial_state(initial_eq)

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)

# ── Descarga de datos ─────────────────────────────────────────────────────────
def get_bitget_candles(client, symbol, granularity, limit=250):
    """
    Descarga velas de Bitget. Devuelve DataFrame con columnas OHLCV.
    granularity: '1H', '4H', '1D', etc.
    """
    try:
        data = client._get("/api/v2/mix/market/candles", {
            "symbol": symbol,
            "productType": "USDT-FUTURES",
            "granularity": granularity,
            "limit": str(limit)
        })
        rows = data.get("data", [])
        if not rows:
            return pd.DataFrame()
        # Bitget devuelve: [timestamp, open, high, low, close, volume, ...]
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "vol", "quote_vol"])
        df["ts"] = pd.to_datetime(df["ts"].astype(float), unit="ms", utc=True)
        df = df.set_index("ts").sort_index()
        for col in ["open", "high", "low", "close", "vol"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["close"])
    except Exception as e:
        print(f"[E11] Error descargando velas {symbol} {granularity}: {e}")
        return pd.DataFrame()

def get_fear_greed(last_known=None):
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1&format=json", timeout=8)
        val = float(r.json()["data"][0]["value"])
        print(f"[E11] Fear & Greed: {val}")
        return val / 100.0
    except Exception as e:
        print(f"[E11] Fear & Greed fallo: {e}. Usando ultimo conocido.")
        return last_known if last_known is not None else 0.5

def download_macro_yfinance():
    """
    Descarga datos diarios de Gold, SPY y DXY con yfinance.
    Devuelve DataFrames o None si falla.
    """
    macro = {}
    for ticker, key in [("GC=F", "gold"), ("SPY", "spy"), ("DX-Y.NYB", "dxy")]:
        try:
            df = yf.download(ticker, period="60d", interval="1d",
                             auto_adjust=False, progress=False)
            if df.empty:
                macro[key] = None
                continue
            # Limpiar columnas MultiIndex si las hay
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            s = pd.to_numeric(df[close_col], errors="coerce").dropna()
            s.index = pd.to_datetime(s.index, utc=True)
            macro[key] = s
            print(f"[E11] {ticker}: {len(s)} dias OK")
        except Exception as e:
            print(f"[E11] {ticker} fallo: {e}")
            macro[key] = None
    return macro

# ── Calcular features ─────────────────────────────────────────────────────────
def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calc_features(df_btc, df_eth, macro, fg_norm):
    """
    Calcula las 26 features sobre el ultimo cierre de df_btc 1H.
    Devuelve Series con los valores o None si hay error.
    """
    df = df_btc.copy()
    c  = df["close"]
    h  = df["high"]
    l  = df["low"]
    v  = df["vol"]

    # Returns
    log_ret_1h = np.log(c / c.shift(1))
    df["log_ret_1h"] = log_ret_1h
    df["log_ret_4h"] = log_ret_1h.rolling(4).sum()
    df["log_ret_1d"] = log_ret_1h.rolling(24).sum()

    # ETH
    if df_eth is not None and not df_eth.empty:
        eth_aligned = df_eth["close"].reindex(df.index, method="ffill")
        df["log_ret_eth"] = np.log(eth_aligned / eth_aligned.shift(1))
        df["rsi_eth"]     = rsi(eth_aligned, 14)
        log_ret_eth_s     = df["log_ret_eth"]
        df["corr_btc_eth_24h"] = log_ret_1h.rolling(24).corr(log_ret_eth_s)
    else:
        df["log_ret_eth"]      = 0.0
        df["rsi_eth"]          = 50.0
        df["corr_btc_eth_24h"] = 0.5

    # Macro (daily → ffill a 1h)
    for key, col in [("gold", "log_ret_gold"), ("dxy", "log_ret_dxy")]:
        if macro.get(key) is not None:
            s = macro[key].reindex(df.index, method="ffill")
            df[col] = np.log(s / s.shift(1)).fillna(0)
        else:
            df[col] = 0.0

    # EMAs
    ema21  = c.ewm(span=21, adjust=False).mean()
    ema50  = c.ewm(span=50, adjust=False).mean()
    sma200 = c.rolling(200).mean()
    df["ema21_diff"]  = (c - ema21)  / c
    df["ema50_diff"]  = (c - ema50)  / c
    df["sma200_diff"] = (c - sma200) / c

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line   = (ema12 - ema26) / c
    macd_sig    = macd_line.ewm(span=9, adjust=False).mean()
    df["macd"]        = macd_line
    df["macd_signal"] = macd_sig
    df["macd_hist"]   = macd_line - macd_sig

    # RSI
    df["rsi_14"] = rsi(c, 14)

    # ROC
    df["roc_6"]  = (c - c.shift(6))  / c.shift(6)
    df["roc_12"] = (c - c.shift(12)) / c.shift(12)
    df["roc_24"] = (c - c.shift(24)) / c.shift(24)
    df["mom_12"] = (c - c.shift(12)) / c

    # ATR
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    df["atr_pct"] = atr14 / c

    # Bollinger
    sma20    = c.rolling(20).mean()
    std20    = c.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    df["bb_width"] = (bb_upper - bb_lower) / sma20
    df["bb_pct"]   = (c - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # Vol rolling
    df["vol_roll_24h"] = log_ret_1h.rolling(24).std()

    # OBV
    direction = np.sign(log_ret_1h.fillna(0))
    obv = (v * direction).cumsum()
    obv_mean = obv.rolling(24).mean()
    obv_std  = obv.rolling(24).std()
    df["obv_norm"] = (obv - obv_mean) / (obv_std + 1e-10)

    # Volume z-score
    vol_mean = v.rolling(24).mean()
    vol_std  = v.rolling(24).std()
    df["volume_zscore"] = (v - vol_mean) / (vol_std + 1e-10)

    # Fear & Greed
    df["fear_greed_norm"] = fg_norm

    last = df[FEATURE_COLS].iloc[-1]
    if last.isna().any():
        print(f"[E11] Features con NaN: {last[last.isna()].index.tolist()}")
        last = last.fillna(0)

    return last

# ── Filtros ───────────────────────────────────────────────────────────────────
def get_regime(df_btc_1h):
    """
    Heuristica HMM simplificada basada en retorno y vol rolling 20 dias.
    Usa datos 1h agrupados a dia.
    """
    daily_close = df_btc_1h["close"].resample("1D").last().dropna()
    if len(daily_close) < 22:
        return "SIDEWAYS"
    log_ret = np.log(daily_close / daily_close.shift(1)).dropna()
    ret20   = log_ret.rolling(20).sum().iloc[-1]
    vol20   = log_ret.rolling(20).std().iloc[-1]
    # Percentil de vol sobre toda la serie
    vol_percentile = (log_ret.rolling(20).std().dropna() < vol20).mean()

    if ret20 > 0 and vol_percentile < 0.70:
        return "BULL"
    elif ret20 < 0 and vol_percentile > 0.70:
        return "BEAR"
    else:
        return "SIDEWAYS"

def get_vol_percentile(df_btc_1h):
    """
    Proxy GARCH: percentil de la vol rolling 20d respecto a los ultimos 504 dias.
    """
    daily_close = df_btc_1h["close"].resample("1D").last().dropna()
    if len(daily_close) < 22:
        return 0.5
    log_ret  = np.log(daily_close / daily_close.shift(1)).dropna()
    vol_roll = log_ret.rolling(20).std().dropna()
    if len(vol_roll) < 2:
        return 0.5
    current_vol = vol_roll.iloc[-1]
    pct = (vol_roll.iloc[-min(504, len(vol_roll)):] < current_vol).mean()
    return float(pct)

def get_ema200_slope(df_btc_1h):
    """
    EMA200 sobre cierres diarios. Devuelve pendiente = EMA200[hoy] - EMA200[hoy-5].
    """
    daily_close = df_btc_1h["close"].resample("1D").last().dropna()
    if len(daily_close) < 205:
        return 1.0  # si no hay suficientes datos, asumir alcista
    ema200 = daily_close.ewm(span=EMA_SPAN, adjust=False).mean()
    slope  = float(ema200.iloc[-1] - ema200.iloc[-1 - EMA_SLOPE_DAYS])
    return slope

# ── Modelo XGBoost ────────────────────────────────────────────────────────────
def load_model(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[E11] No se pudo cargar modelo {path}: {e}")
        return None

def predict_proba(model, x):
    """x: array shape (1, 26)"""
    try:
        prob = model.predict_proba(x)[0, 1]
        return float(prob)
    except Exception as e:
        print(f"[E11] Error en predict_proba: {e}")
        return 0.5

# ── Kelly sizing ──────────────────────────────────────────────────────────────
def kelly_sizing(prob):
    edge = 2 * (prob - 0.5)
    pct  = BASE_PCT + edge / KELLY_DIV
    return min(pct, PCT_MAX)

# ── Condiciones de parada ─────────────────────────────────────────────────────
def check_auto_pause(state, equity, reason=None):
    initial = state.get("initial_equity", 0)
    # Drawdown > 50%
    if initial > 0 and equity < initial * 0.50:
        state["paused"]       = True
        state["pause_reason"] = "DRAWDOWN > 50%"
        print("[E11] *** AUTO-PAUSE: drawdown > 50% ***")
        return True
    # 5 perdidas consecutivas
    if state.get("consecutive_losses", 0) >= 5:
        state["paused"]       = True
        state["pause_reason"] = "5 trades negativos consecutivos"
        print("[E11] *** AUTO-PAUSE: 5 perdidas consecutivas ***")
        return True
    # 3 errores API seguidos
    if state.get("consecutive_errors", 0) >= 3:
        state["paused"]       = True
        state["pause_reason"] = "3 errores API consecutivos"
        print("[E11] *** AUTO-PAUSE: 3 errores API consecutivos ***")
        return True
    if reason:
        state["paused"]       = True
        state["pause_reason"] = reason
        return True
    return False

# ── Publicar data.json ────────────────────────────────────────────────────────
def publish_data(state, btc_price, filters, prob):
    trades    = state.get("trades", [])
    wins      = [t for t in trades if t.get("pnl_usdt", 0) > 0]
    win_rate  = round(len(wins) / len(trades) * 100, 1) if trades else 0.0
    total_pnl = round(sum(t.get("pnl_usdt", 0) for t in trades), 4)

    bal = state.get("live_balance", {"equity": 0, "available": 0, "unrealized_pl": 0})

    # PnL diario
    daily_pnl = {}
    for t in trades:
        day = str(t.get("close_time", ""))[:10]
        if not day or day == "None":
            continue
        daily_pnl[day] = round(daily_pnl.get(day, 0) + t.get("pnl_usdt", 0), 4)

    data = {
        "updated_at":        datetime.now(timezone.utc).isoformat(),
        "btc_price":         round(btc_price, 2),
        "runs":              state["runs"],
        "live_mode":         LIVE_MODE,
        "paused":            state.get("paused", False),
        "pause_reason":      state.get("pause_reason"),
        "live_balance": {
            "equity":        round(float(bal.get("equity", 0)), 4),
            "available":     round(float(bal.get("available", 0)), 4),
            "unrealized_pl": round(float(bal.get("unrealized_pl", 0)), 4),
        },
        "initial_equity":    state.get("initial_equity", 0),
        "total_pnl_usdt":    total_pnl,
        "num_trades":        len(trades),
        "win_rate":          win_rate,
        "filters":           filters,
        "last_prob":         round(prob, 4) if prob is not None else None,
        "position":          state.get("position", {}),
        "trades":            trades[-50:],
        "equity_history":    state.get("equity_history", [])[-300:],
        "daily_pnl":         [{"date": k, "pnl": v} for k, v in sorted(daily_pnl.items())],
    }

    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, separators=(",", ":"), default=str)
    print(f"[E11] docs/data.json OK — BTC={btc_price:.0f}  senal={filters.get('signal')}  paused={state.get('paused')}")

# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    now_utc = datetime.now(timezone.utc).isoformat()
    print(f"\n{'='*55}")
    print(f"  E11 Bot — {now_utc}")
    print(f"  Modo: {'LIVE Bitget' if LIVE_MODE else 'PAPER ONLY'}")
    print(f"{'='*55}")

    state = load_state()
    state["runs"] = state.get("runs", 0) + 1
    state["last_run"] = now_utc

    # Verificar si el bot esta pausado
    if state.get("paused"):
        print(f"[E11] Bot PAUSADO — razon: {state.get('pause_reason')}")
        save_state(state)
        # Publicar estado aunque este pausado
        try:
            btc_price = 0
            if LIVE_MODE:
                client = client_from_env()
                btc_price = client.get_price("BTCUSDT")
            publish_data(state, btc_price, {"signal": "PAUSED"}, None)
        except Exception:
            pass
        return

    # Inicializar cliente Bitget
    client = None
    if LIVE_MODE:
        try:
            client = client_from_env()
        except Exception as e:
            print(f"[E11] Error inicializando cliente: {e}")
            state["consecutive_errors"] = state.get("consecutive_errors", 0) + 1
            check_auto_pause(state, 0, f"Error API: {e}")
            save_state(state)
            return

    # Actualizar balance desde Bitget
    balance = {"available": 0, "equity": 0, "unrealized_pl": 0}
    if client:
        try:
            balance = client.get_balance()
            state["live_balance"] = balance
            print(f"[E11] Balance: equity={balance['equity']:.4f}  disponible={balance['available']:.4f}")
            # Inicializar equity inicial si es la primera vez
            if state.get("initial_equity", 0) == 0:
                state["initial_equity"] = round(balance["equity"], 4)
                print(f"[E11] Equity inicial establecido: {state['initial_equity']:.4f}")
        except Exception as e:
            print(f"[E11] Error leyendo balance: {e}")
            state["consecutive_errors"] = state.get("consecutive_errors", 0) + 1

    equity = float(balance.get("equity", 0))
    if check_auto_pause(state, equity):
        save_state(state)
        return

    # Precio actual
    btc_price = 0
    if client:
        try:
            btc_price = client.get_price("BTCUSDT")
            print(f"[E11] BTC precio: {btc_price:.2f}")
        except Exception as e:
            print(f"[E11] Error obteniendo precio: {e}")

    # ── Verificar si hay posicion abierta que cerrar ──────────────────────────
    pos = state.get("position", {})
    if pos.get("open") and client:
        # Comprobar en Bitget si la posicion sigue abierta (puede haberse cerrado por SL/TP)
        try:
            has_pos = client.has_open_position("BTCUSDT")
            if not has_pos:
                print("[E11] Posicion cerrada externamente (SL/TP alcanzado)")
                # Registrar trade con precio actual
                entry = pos.get("entry_price", btc_price)
                size  = pos.get("size_btc", 0)
                pnl   = (btc_price - entry) * size * (1 if pos.get("side") == "LONG" else -1)
                pnl_net = pnl - 2 * COMISION * pos.get("size_usdt", 0)
                trade_record = {
                    "side":       pos.get("side"),
                    "entry_price": round(entry, 2),
                    "exit_price":  round(btc_price, 2),
                    "size_btc":    pos.get("size_btc"),
                    "size_usdt":   pos.get("size_usdt"),
                    "pct_used":    pos.get("pct_used"),
                    "leverage":    pos.get("leverage"),
                    "pnl_usdt":    round(pnl_net, 4),
                    "pnl_pct":     round(pnl_net / pos.get("size_usdt", 1) * 100, 2),
                    "open_time":   pos.get("open_time"),
                    "close_time":  now_utc,
                    "exit_reason": "SL_TP_OR_EXTERNAL"
                }
                state.setdefault("trades", []).append(trade_record)
                state["position"] = {"open": False}
                # Actualizar racha de perdidas
                if pnl_net < 0:
                    state["consecutive_losses"] = state.get("consecutive_losses", 0) + 1
                else:
                    state["consecutive_losses"] = 0
                print(f"[E11] Trade registrado: PnL={pnl_net:.4f} USDT")
        except Exception as e:
            print(f"[E11] Error verificando posicion: {e}")
            state["consecutive_errors"] = state.get("consecutive_errors", 0) + 1

    # ── Descargar datos de mercado ────────────────────────────────────────────
    print("[E11] Descargando datos 1h BTC y ETH...")
    df_btc = pd.DataFrame()
    df_eth = pd.DataFrame()

    if client:
        df_btc = get_bitget_candles(client, "BTCUSDT", "1H", limit=250)
        df_eth = get_bitget_candles(client, "ETHUSDT", "1H", limit=250)
    else:
        # Fallback yfinance para paper mode
        try:
            raw = yf.download("BTC-USD", period="30d", interval="1h",
                              auto_adjust=False, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.index = pd.to_datetime(raw.index, utc=True)
            df_btc = raw.rename(columns={"Open":"open","High":"high","Low":"low",
                                          "Close":"close","Volume":"vol"})
            btc_price = float(df_btc["close"].iloc[-1])
        except Exception as e:
            print(f"[E11] Error descargando datos fallback: {e}")

    if df_btc.empty:
        print("[E11] Sin datos BTC — abortando")
        state["consecutive_errors"] = state.get("consecutive_errors", 0) + 1
        check_auto_pause(state, equity)
        save_state(state)
        return

    state["consecutive_errors"] = 0
    btc_price = float(df_btc["close"].iloc[-1])
    print(f"[E11] BTC velas: {len(df_btc)}  precio cierre: {btc_price:.2f}")

    # Macro yfinance
    print("[E11] Descargando macro (Gold, DXY)...")
    macro = download_macro_yfinance()

    # Fear & Greed
    last_fg = state.get("last_filters", {}).get("fear_greed_raw")
    fg_norm = get_fear_greed(last_fg)

    # ── Calcular features ──────────────────────────────────────────────────────
    print("[E11] Calculando 26 features...")
    try:
        features = calc_features(df_btc, df_eth if not df_eth.empty else None, macro, fg_norm)
        x = features.values.reshape(1, 26)
    except Exception as e:
        print(f"[E11] Error calculando features: {e}")
        save_state(state)
        return

    # ── Pipeline de filtros ────────────────────────────────────────────────────
    signal = "HOLD"
    prob   = None
    filters = {}

    # Filtro 1: regimen HMM heuristico
    regime = get_regime(df_btc)
    filters["regime"] = regime
    print(f"[E11] Filtro 1 — Regimen: {regime}")

    # Filtro 2: volatilidad GARCH proxy
    vol_pct = get_vol_percentile(df_btc)
    filters["vol_percentile"] = round(vol_pct, 4)
    vol_ok = vol_pct <= GARCH_VOL_UMBRAL
    filters["vol_ok"] = vol_ok
    print(f"[E11] Filtro 2 — Vol percentil: {vol_pct:.3f}  OK={vol_ok}")

    # Filtro 3: tendencia EMA200
    ema_slope = get_ema200_slope(df_btc)
    filters["ema200_slope"] = round(ema_slope, 2)
    ema_ok = ema_slope > 0
    filters["ema200_ok"] = ema_ok
    print(f"[E11] Filtro 3 — EMA200 slope: {ema_slope:.2f}  OK={ema_ok}")

    # Filtro 4: XGBoost — solo si estamos en BULL y los demas filtros pasan
    if regime == "BULL" and vol_ok and ema_ok:
        model = load_model("models/xgb_bull.pkl")
        if model is None:
            check_auto_pause(state, equity, "Modelo xgb_bull.pkl no cargable")
            save_state(state)
            return
        prob = predict_proba(model, x)
        filters["prob"] = round(prob, 4)
        in_dead_zone = abs(prob - 0.5) <= DELTA
        filters["in_dead_zone"] = in_dead_zone
        print(f"[E11] Filtro 4 — XGBoost prob: {prob:.4f}  dead_zone={in_dead_zone}")

        if not in_dead_zone and prob > 0.5:
            signal = "LONG"
    elif regime != "BULL":
        print("[E11] No BULL — solo opera en regimen BULL")
    elif not vol_ok:
        print(f"[E11] Volatilidad demasiado alta ({vol_pct:.3f} > {GARCH_VOL_UMBRAL})")
    elif not ema_ok:
        print(f"[E11] EMA200 bajista (slope={ema_slope:.2f}) — no operar LONG")

    filters["signal"] = signal
    filters["fear_greed_raw"] = round(fg_norm * 100, 1)
    state["last_filters"] = filters
    state["last_signal"]  = signal

    print(f"\n[E11] *** SENAL FINAL: {signal} ***\n")

    # ── Ejecutar trade ─────────────────────────────────────────────────────────
    if signal == "LONG" and not state.get("position", {}).get("open"):
        pct        = kelly_sizing(prob)
        capital    = float(balance.get("available", equity))
        size_usdt  = capital * pct
        filters["pct_kelly"] = round(pct, 4)
        print(f"[E11] Kelly sizing: pct={pct:.3f}  capital={capital:.2f}  size={size_usdt:.2f} USDT")

        if client and size_usdt >= 5:
            try:
                # TP: +3% sobre precio, SL: -2% sobre precio
                tp_price = btc_price * 1.03
                sl_price = btc_price * 0.98
                result = client.place_order(
                    symbol    = "BTCUSDT",
                    direction = "buy",
                    size_usdt = size_usdt * LEVERAGE,
                    sl_price  = sl_price,
                    tp_price  = tp_price,
                    leverage  = LEVERAGE
                )
                state["position"] = {
                    "open":        True,
                    "side":        "LONG",
                    "entry_price": result["entry_px"],
                    "size_btc":    result["qty"],
                    "size_usdt":   round(size_usdt, 4),
                    "pct_used":    round(pct, 4),
                    "leverage":    LEVERAGE,
                    "open_time":   now_utc,
                    "order_id":    result["orderId"],
                    "sl_price":    round(sl_price, 2),
                    "tp_price":    round(tp_price, 2),
                }
                state["consecutive_errors"] = 0
                print(f"[E11] LONG abierto: {result['qty']} BTC @ {result['entry_px']:.2f}  SL={sl_price:.0f}  TP={tp_price:.0f}")
            except Exception as e:
                print(f"[E11] Error abriendo posicion: {e}")
                state["consecutive_errors"] = state.get("consecutive_errors", 0) + 1
                check_auto_pause(state, equity)
        elif not client:
            print(f"[E11] Paper mode — LONG simulado: {size_usdt:.2f} USDT")
            state["position"] = {
                "open":        True,
                "side":        "LONG",
                "entry_price": btc_price,
                "size_btc":    round(size_usdt / btc_price, 6),
                "size_usdt":   round(size_usdt, 4),
                "pct_used":    round(pct, 4),
                "leverage":    LEVERAGE,
                "open_time":   now_utc,
                "order_id":    "PAPER",
                "sl_price":    round(btc_price * 0.98, 2),
                "tp_price":    round(btc_price * 1.03, 2),
            }
        else:
            print(f"[E11] size_usdt={size_usdt:.2f} demasiado pequeno (<5 USDT), no se ejecuta")

    elif signal == "HOLD" and state.get("position", {}).get("open"):
        print("[E11] Senal HOLD con posicion abierta — mantener hasta SL/TP")

    # ── Equity history ────────────────────────────────────────────────────────
    eq_entry = {
        "ts":        now_utc,
        "equity":    round(equity, 4),
        "unrealPL":  round(float(balance.get("unrealized_pl", 0)), 4)
    }
    state.setdefault("equity_history", []).append(eq_entry)
    # Limitar a 2000 entradas (~33h a 1 min)
    if len(state["equity_history"]) > 2000:
        state["equity_history"] = state["equity_history"][-2000:]

    save_state(state)
    publish_data(state, btc_price, filters, prob)
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run()
