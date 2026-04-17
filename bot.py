"""
bot.py — E11 XGBoost Bot
=========================
Pipeline: descarga datos 1h → calcula 26 features → filtros HMM/GARCH/dead zone → XGBoost → E11/E6 → ejecuta en Bitget
Se ejecuta cada minuto via GitHub Actions. Intenta E11 primero y, si no pasa, intenta E6.
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
DELTA_GLOBAL    = 0.05    # dead zone global previa (filtro 3 del pipeline TFG)
DELTA           = 0.20    # dead zone E11 especifica
LEVERAGE        = 5       # apalancamiento fijo
BASE_PCT        = 0.20    # sizing minimo (20% del capital)
PCT_MAX         = 0.55    # sizing maximo (55% del capital)
KELLY_DIV       = 3.0     # divisor Kelly fraccional
GARCH_VOL_UMBRAL= 0.80    # percentil de corte de volatilidad
EMA_SPAN        = 200     # periodo EMA tendencia diaria
EMA_SLOPE_DAYS  = 5       # ventana pendiente EMA200
ALLOW_SHORTS    = False   # shorts desactivados en E11
COMISION        = 0.0004  # 0.04% por lado

# ── Parametros estrategia E6 (fallback) ──────────────────────────────────────
E6_DELTA   = 0.30    # dead zone E6 — mas selectivo
E6_PCT     = 0.40    # sizing fijo 40% del capital
# E6 opera en cualquier regimen (BULL, BEAR, SIDEWAYS) con LONG y SHORT
# usando el submodelo XGBoost correspondiente al regimen HMM del momento.

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
        try:
            with open(STATE_FILE) as f:
                content = f.read().strip()
            if content:
                return json.loads(content)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[E11] state.json corrupto ({e}) — reiniciando estado")
    # Si no existe o esta corrupto, leer equity de Bitget
    print("[E11] Inicializando state.json...")
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
    """Kelly fraccional de E11 segun el notebook sintetizado."""
    edge = 2 * abs(prob - 0.5)
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
        "active_strategy":   filters.get("strategy", "NONE"),
        "position":          state.get("position", {}),
        "trades":            trades[-50:],
        "equity_history":    state.get("equity_history", [])[-300:],
        "prob_history":      state.get("prob_history", [])[-300:],
        "regime_history":    state.get("regime_history", [])[-1440:],
        "candles_1h":        state.get("candles_1h", []),
        "daily_pnl":         [{"date": k, "pnl": v} for k, v in sorted(daily_pnl.items())],
    }

    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, separators=(",", ":"), default=str)
    print(f"[E11] docs/data.json OK — BTC={btc_price:.0f}  senal={filters.get('signal')}  paused={state.get('paused')}")

# ── Registrar cierre de trade ─────────────────────────────────────────────────
def _register_trade_close(state, pos, exit_price, close_time, exit_reason):
    """
    Registra un trade cerrado en state["trades"] y limpia la posicion.
    Aplica comision por ambos lados (apertura + cierre) como en el backtest.
    """
    entry = pos.get("entry_price", exit_price)
    size_btc = pos.get("size_btc", 0)
    size_usdt = pos.get("size_usdt", 0)
    leverage = pos.get("leverage", 1)
    side = pos.get("side", "LONG")

    direction_sign = 1 if side == "LONG" else -1
    ret = (exit_price - entry) / entry
    pnl_bruto = size_usdt * leverage * direction_sign * ret
    comision_total = size_usdt * (COMISION * 2)  # apertura + cierre
    pnl_net = pnl_bruto - comision_total

    # Cap de perdida al 90% del margen (igual que en el backtest)
    if pnl_net < -0.90 * size_usdt:
        pnl_net = -0.90 * size_usdt

    trade_record = {
        "side":        side,
        "strategy":    pos.get("strategy"),
        "entry_price": round(entry, 2),
        "exit_price":  round(exit_price, 2),
        "size_btc":    size_btc,
        "size_usdt":   size_usdt,
        "pct_used":    pos.get("pct_used"),
        "leverage":    leverage,
        "pnl_usdt":    round(pnl_net, 4),
        "pnl_pct":     round(pnl_net / size_usdt * 100, 2) if size_usdt else 0,
        "open_time":   pos.get("open_time"),
        "close_time":  close_time,
        "open_candle_ts":  pos.get("open_candle_ts"),
        "exit_reason": exit_reason,
    }
    state.setdefault("trades", []).append(trade_record)
    state["position"] = {"open": False}

    if pnl_net < 0:
        state["consecutive_losses"] = state.get("consecutive_losses", 0) + 1
    else:
        state["consecutive_losses"] = 0

    print(f"[E11] Trade cerrado ({exit_reason}): {side} entry={entry:.0f} exit={exit_price:.0f} PnL={pnl_net:.4f} USDT")


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
    # Timestamp (ISO) de la ultima vela 1h cerrada — nuestro "reloj" para el cierre
    last_candle_ts = df_btc.index[-1].isoformat()
    print(f"[E11] BTC velas: {len(df_btc)}  precio cierre: {btc_price:.2f}  vela actual: {last_candle_ts}")

    # Snapshot de las ultimas 240 velas 1h (10 dias) para el grafico del dashboard
    candles_snapshot = []
    for ts_c, row_c in df_btc.tail(240).iterrows():
        candles_snapshot.append({
            "ts":    ts_c.isoformat(),
            "open":  round(float(row_c["open"]), 2),
            "high":  round(float(row_c["high"]), 2),
            "low":   round(float(row_c["low"]), 2),
            "close": round(float(row_c["close"]), 2),
        })
    state["candles_1h"] = candles_snapshot

    # ── Verificar si hay posicion abierta que cerrar ──────────────────────────
    # Regla del backtest: cerrar en el close de la vela SIGUIENTE a la de apertura.
    # Es decir, cerramos en cuanto last_candle_ts != open_candle_ts.
    pos = state.get("position", {})
    if pos.get("open"):
        open_candle_ts = pos.get("open_candle_ts")

        if client:
            # Comprobar que la posicion sigue abierta en Bitget
            try:
                has_pos = client.has_open_position("BTCUSDT")
            except Exception as e:
                print(f"[E11] Error verificando posicion: {e}")
                state["consecutive_errors"] = state.get("consecutive_errors", 0) + 1
                has_pos = True  # asumir que sigue abierta para no hacer nada raro

            if not has_pos:
                # La posicion ya no existe en Bitget (liquidacion, cierre manual, etc.)
                print("[E11] Posicion no encontrada en Bitget — registrando cierre externo")
                _register_trade_close(state, pos, btc_price, now_utc,
                                      exit_reason="EXTERNAL_CLOSE")
            elif open_candle_ts and last_candle_ts != open_candle_ts:
                # Ha cambiado la vela 1h → cierre forzoso a mercado (replica backtest)
                print(f"[E11] Vela 1h cambio ({open_candle_ts} -> {last_candle_ts}) — cerrando a mercado")
                try:
                    client.close_position("BTCUSDT")
                    time.sleep(2)  # esperar a que se procese el cierre
                    exit_px = client.get_price("BTCUSDT")
                    _register_trade_close(state, pos, exit_px, now_utc,
                                          exit_reason="CANDLE_CLOSE")
                except Exception as e:
                    print(f"[E11] Error cerrando posicion: {e}")
                    state["consecutive_errors"] = state.get("consecutive_errors", 0) + 1
            else:
                print(f"[E11] Posicion abierta en vela {open_candle_ts}, aun es la vela actual — mantener")
        else:
            # Paper mode: cerrar si ha cambiado la vela
            if open_candle_ts and last_candle_ts != open_candle_ts:
                print(f"[E11] (paper) Vela 1h cambio — cerrando simulado")
                _register_trade_close(state, pos, btc_price, now_utc,
                                      exit_reason="CANDLE_CLOSE_PAPER")

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
    signal   = "HOLD"
    prob     = None
    strategy = None   # "E11" o "E6" — cual estrategia genero la senal
    filters  = {}

    # Filtros comunes a ambas estrategias
    regime    = get_regime(df_btc)
    vol_pct   = get_vol_percentile(df_btc)
    ema_slope = get_ema200_slope(df_btc)
    vol_ok    = vol_pct <= GARCH_VOL_UMBRAL
    ema_ok    = ema_slope > 0

    filters["regime"]        = regime
    filters["vol_percentile"]= round(vol_pct, 4)
    filters["vol_ok"]        = vol_ok
    filters["ema200_slope"]  = round(ema_slope, 2)
    filters["ema200_ok"]     = ema_ok

    print(f"[BOT] Regimen: {regime}  |  Vol: {vol_pct:.3f}  |  EMA200 slope: {ema_slope:.2f}")

    # ── Seleccion de submodelo segun regimen HMM (igual que pipeline TFG) ──────
    # BULL -> xgb_bull.pkl    |    BEAR/SIDEWAYS -> xgb_bear.pkl
    if regime == "BULL":
        model_active = load_model("models/xgb_bull.pkl")
        model_tag = "xgb_bull"
    else:
        model_active = load_model("models/xgb_bear.pkl")
        model_tag = "xgb_bear"

    if model_active is None:
        check_auto_pause(state, equity, f"Modelo {model_tag}.pkl no cargable")
        save_state(state)
        return

    # Calcular probabilidad (una sola vez, la misma para E11 y E6)
    xgb_prob = predict_proba(model_active, x)
    filters["prob_e11"] = round(xgb_prob, 4)
    filters["prob_e6"]  = round(xgb_prob, 4)
    print(f"[BOT] XGBoost ({model_tag}) prob: {xgb_prob:.4f}")

    # ── Filtro 3 del pipeline TFG: dead zone global 0.05 ───────────────────────
    in_dead_zone_global = abs(xgb_prob - 0.5) <= DELTA_GLOBAL
    filters["in_dead_zone_global"] = in_dead_zone_global

    # ── Intentar E11 primero ──────────────────────────────────────────────────
    # Reglas del TFG (run_backtest_e11):
    #   - active = vol_ok + |prob-0.5| > 0.05
    #   - |prob-0.5| > DELTA_E11 (=0.20)
    #   - shorts desactivados (ALLOW_SHORTS=False)
    #   - macro_up (EMA200 slope > 0) solo para LONG
    # NO exige regime == BULL (el regimen solo elige submodelo, no bloquea)
    print("[BOT] Intentando E11 (macro + Kelly + shorts off)...")
    in_dead_zone_e11 = abs(xgb_prob - 0.5) <= DELTA
    filters["in_dead_zone_e11"] = in_dead_zone_e11

    if not vol_ok:
        print(f"[E11] Bloqueado: vol_pct={vol_pct:.3f} > {GARCH_VOL_UMBRAL}")
    elif in_dead_zone_global:
        print(f"[E11] Bloqueado: dead zone global |{xgb_prob:.3f}-0.5| <= {DELTA_GLOBAL}")
    elif in_dead_zone_e11:
        print(f"[E11] Bloqueado: dead zone E11 |{xgb_prob:.3f}-0.5| <= {DELTA}")
    else:
        direccion = 1 if xgb_prob > 0.5 else -1
        if direccion == -1 and not ALLOW_SHORTS:
            print(f"[E11] Bloqueado: prob={xgb_prob:.3f}<0.5 y shorts desactivados")
        elif direccion == 1 and not ema_ok:
            print(f"[E11] Bloqueado: LONG requiere macro_up pero EMA200 slope={ema_slope:.2f}<=0")
        else:
            # Senal valida
            if direccion == 1:
                signal   = "LONG"
                strategy = "E11"
                prob     = xgb_prob
                print(f"[E11] *** SENAL LONG — prob={xgb_prob:.4f}  regimen={regime} ***")
            else:
                signal   = "SHORT"
                strategy = "E11"
                prob     = xgb_prob
                print(f"[E11] *** SENAL SHORT — prob={xgb_prob:.4f}  regimen={regime} ***")

    # ── Si E11 no da senal, intentar E6 como fallback ─────────────────────────
    # Reglas del TFG (run_backtest con delta=0.30, apal=5, pct=0.40):
    #   - active = vol_ok + |prob-0.5| > 0.05
    #   - |prob-0.5| > DELTA_E6 (=0.30)
    #   - LONG si prob>0.5 / SHORT si prob<0.5  (en cualquier regimen)
    if signal == "HOLD" and not state.get("position", {}).get("open"):
        print("[BOT] E11 sin senal — intentando E6 (delta=0.30, LONG+SHORT)...")
        in_dead_zone_e6 = abs(xgb_prob - 0.5) <= E6_DELTA
        filters["in_dead_zone_e6"] = in_dead_zone_e6

        if not vol_ok:
            print(f"[E6] Bloqueado: vol_pct={vol_pct:.3f} > {GARCH_VOL_UMBRAL}")
        elif in_dead_zone_global:
            print(f"[E6] Bloqueado: dead zone global |{xgb_prob:.3f}-0.5| <= {DELTA_GLOBAL}")
        elif in_dead_zone_e6:
            print(f"[E6] Bloqueado: dead zone E6 |{xgb_prob:.3f}-0.5| <= {E6_DELTA}")
        else:
            if xgb_prob > 0.5:
                signal   = "LONG"
                strategy = "E6"
                prob     = xgb_prob
                print(f"[E6] *** SENAL LONG — prob={xgb_prob:.4f}  regimen={regime} ***")
            else:
                signal   = "SHORT"
                strategy = "E6"
                prob     = xgb_prob
                print(f"[E6] *** SENAL SHORT — prob={xgb_prob:.4f}  regimen={regime} ***")

    filters["signal"]        = signal
    filters["strategy"]      = strategy if strategy else "NONE"
    filters["fear_greed_raw"]= round(fg_norm * 100, 1)
    state["last_filters"]    = filters
    state["last_signal"]     = signal

    # Historial de probabilidades (para grafico de confianza en dashboard)
    prob_e11 = filters.get("prob_e11")
    prob_e6  = filters.get("prob_e6")
    prob_entry = {
        "ts":       now_utc,
        "regime":   regime,
        "prob_e11": round(prob_e11, 4) if prob_e11 is not None else None,
        "prob_e6":  round(prob_e6, 4)  if prob_e6  is not None else None,
        "signal":   signal,
        "strategy": strategy if strategy else "NONE",
    }
    state.setdefault("prob_history", []).append(prob_entry)
    # Limitar a 500 entradas (~8h a 1 min)
    if len(state["prob_history"]) > 500:
        state["prob_history"] = state["prob_history"][-500:]

    # Historial de regimen HMM por vela 1h (para bandas de regimen en dashboard).
    # Solo anadimos una entrada cuando cambia la vela; varias ejecuciones del bot
    # dentro de la misma vela no duplican datos.
    rh = state.setdefault("regime_history", [])
    if not rh or rh[-1].get("ts") != last_candle_ts:
        rh.append({"ts": last_candle_ts, "regime": regime})
        # Mantener ~60 dias a 1h = 1440 velas
        if len(rh) > 1440:
            state["regime_history"] = rh[-1440:]

    print(f"\n[BOT] *** SENAL FINAL: {signal}  ESTRATEGIA: {strategy or 'NINGUNA'} ***\n")

    # ── Ejecutar trade ─────────────────────────────────────────────────────────
    if signal in ("LONG", "SHORT") and not state.get("position", {}).get("open"):
        capital   = float(balance.get("available", equity))
        direction = "buy" if signal == "LONG" else "sell"

        # Sizing segun estrategia
        if strategy == "E11":
            pct       = kelly_sizing(prob)
            size_usdt = capital * pct
            filters["pct_kelly"] = round(pct, 4)
            print(f"[E11] Kelly sizing: pct={pct:.3f}  capital={capital:.2f}  size={size_usdt:.2f} USDT")
        else:
            # E6: sizing fijo 40%
            pct       = E6_PCT
            size_usdt = capital * pct
            filters["pct_e6"] = round(pct, 4)
            print(f"[E6] Sizing fijo: pct={pct:.2f}  capital={capital:.2f}  size={size_usdt:.2f} USDT")

        if client and size_usdt >= 5:
            try:
                result = client.place_order(
                    symbol    = "BTCUSDT",
                    direction = direction,
                    size_usdt = size_usdt * LEVERAGE,
                    leverage  = LEVERAGE
                )
                state["position"] = {
                    "open":           True,
                    "side":           signal,
                    "strategy":       strategy,
                    "entry_price":    result["entry_px"],
                    "size_btc":       result["qty"],
                    "size_usdt":      round(size_usdt, 4),
                    "pct_used":       round(pct, 4),
                    "leverage":       LEVERAGE,
                    "open_time":      now_utc,
                    "open_candle_ts": last_candle_ts,   # << clave para cierre a siguiente vela
                    "order_id":       result["orderId"],
                }
                state["consecutive_errors"] = 0
                print(f"[{strategy}] {signal} abierto: {result['qty']} BTC @ {result['entry_px']:.2f}  "
                      f"vela={last_candle_ts} — se cerrara al cambiar de vela 1h")
            except Exception as e:
                print(f"[BOT] Error abriendo posicion: {e}")
                state["consecutive_errors"] = state.get("consecutive_errors", 0) + 1
                check_auto_pause(state, equity)
        elif not client:
            print(f"[{strategy}] Paper mode — {signal} simulado: {size_usdt:.2f} USDT")
            state["position"] = {
                "open":           True,
                "side":           signal,
                "strategy":       strategy,
                "entry_price":    btc_price,
                "size_btc":       round(size_usdt / btc_price, 6),
                "size_usdt":      round(size_usdt, 4),
                "pct_used":       round(pct, 4),
                "leverage":       LEVERAGE,
                "open_time":      now_utc,
                "open_candle_ts": last_candle_ts,
                "order_id":       "PAPER",
            }
        else:
            print(f"[BOT] size_usdt={size_usdt:.2f} demasiado pequeno (<5 USDT), no se ejecuta")

    elif signal == "HOLD" and state.get("position", {}).get("open"):
        print("[BOT] Senal HOLD con posicion abierta — mantener hasta cambio de vela")

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
