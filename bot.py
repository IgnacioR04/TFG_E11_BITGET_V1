from pathlib import Path

path = Path("bot.py")
text = path.read_text(encoding="utf-8")

old = """\"\"\"
bot.py — E11 XGBoost Bot
=========================
Pipeline: descarga datos 1h → calcula 26 features → filtros HMM/GARCH/EMA200 → XGBoost → Kelly sizing → ejecuta en Bitget
Se ejecuta cada minuto via GitHub Actions. Solo opera LONG cuando todos los filtros pasan.
La posicion se abre y cierra en la misma hora (holding = 1 ciclo de senal).
\"\"\""""
new = """\"\"\"
bot.py — BTC live bot con cascada E11 -> E6
===========================================
Pipeline live
1. Descarga contexto 1h por yfinance
2. Reconstruye la vela actual con 1h + 1m para aproximar modo live
3. Calcula features, HMM diario, proxy de volatilidad y EMA200 slope
4. Evalua primero E11
5. Si E11 no entra, evalua E6
6. Si ninguna entra, no opera

Reglas clave
- E11 tiene prioridad absoluta
- E11 solo LONG
- E11 requiere regimen BULL
- E11 requiere EMA200 slope > 0
- E6 actua solo como fallback
- La posicion se cierra al cambiar la vela 1h siguiente a la de apertura
\"\"\""""
text = text.replace(old, new)

anchor = """def download_hourly_yfinance(ticker, period=YF_SIGNAL_PERIOD):
    try:
        raw = yf.download(ticker, period=period, interval="1h",
                          auto_adjust=False, progress=False)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
        raw.index = pd.to_datetime(raw.index, utc=True)
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
        out = raw[keep].copy()
        if "volume" in out.columns:
            out = out.rename(columns={"volume": "vol"})
        return out.dropna(subset=["close"]).sort_index()
    except Exception as e:
        print(f"[E11] yfinance {ticker} fallo: {e}")
        return pd.DataFrame()
"""
add = """
def download_minute_yfinance(ticker, period="7d"):
    try:
        raw = yf.download(
            ticker,
            period=period,
            interval="1m",
            auto_adjust=False,
            progress=False,
            prepost=False,
        )
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
        raw.index = pd.to_datetime(raw.index, utc=True)
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
        out = raw[keep].copy()
        if "volume" in out.columns:
            out = out.rename(columns={"volume": "vol"})
        return out.dropna(subset=["close"]).sort_index()
    except Exception as e:
        print(f"[E11] yfinance 1m {ticker} fallo: {e}")
        return pd.DataFrame()


def build_live_hour_from_minutes(df_1m):
    if df_1m is None or df_1m.empty:
        return None, None
    current_hour = df_1m.index[-1].floor("1h")
    chunk = df_1m[df_1m.index >= current_hour].copy()
    if chunk.empty:
        return None, None
    row = pd.Series({
        "open": float(chunk["open"].iloc[0]),
        "high": float(chunk["high"].max()),
        "low": float(chunk["low"].min()),
        "close": float(chunk["close"].iloc[-1]),
        "vol": float(chunk["vol"].sum()),
    })
    return current_hour, row


def merge_hourly_with_live(df_1h, df_1m):
    if df_1h is None or df_1h.empty:
        return df_1h, False
    if df_1m is None or df_1m.empty:
        return df_1h.sort_index(), False

    live_ts, live_row = build_live_hour_from_minutes(df_1m)
    if live_ts is None:
        return df_1h.sort_index(), False

    out = df_1h.copy().sort_index()
    if live_ts in out.index:
        out = out.drop(index=live_ts)
    out.loc[live_ts, ["open", "high", "low", "close", "vol"]] = [
        live_row["open"],
        live_row["high"],
        live_row["low"],
        live_row["close"],
        live_row["vol"],
    ]
    out = out.sort_index()
    return out, True


def build_live_market_data(ticker, minute_ticker=None, period_1h=YF_SIGNAL_PERIOD):
    base = download_hourly_yfinance(ticker, period=period_1h)
    df_1m = download_minute_yfinance(minute_ticker or ticker)
    merged, used_live = merge_hourly_with_live(base, df_1m)
    return merged, df_1m, used_live


def empty_position():
    return {
        "open": False,
        "side": None,
        "strategy": None,
        "entry_price": None,
        "size_btc": None,
        "size_usdt": None,
        "pct_used": None,
        "leverage": None,
        "open_time": None,
        "open_candle_ts": None,
        "order_id": None,
    }
"""
text = text.replace(anchor, anchor + add)

text = text.replace(
"""        "position": {
            "open": False,
            "side": None,
            "entry_price": None,
            "size_btc": None,
            "size_usdt": None,
            "pct_used": None,
            "leverage": None,
            "open_time": None,
            "order_id": None
        },""",
"""        "position": empty_position(),"""
)

old_publish = """    data = {
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
    }"""
new_publish = """    data = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "btc_price": round(btc_price, 2),
        "runs": state["runs"],
        "live_mode": LIVE_MODE,
        "paused": state.get("paused", False),
        "pause_reason": state.get("pause_reason"),
        "live_balance": {
            "equity": round(float(bal.get("equity", 0)), 4),
            "available": round(float(bal.get("available", 0)), 4),
            "unrealized_pl": round(float(bal.get("unrealized_pl", 0)), 4),
        },
        "initial_equity": state.get("initial_equity", 0),
        "total_pnl_usdt": total_pnl,
        "num_trades": len(trades),
        "win_rate": win_rate,
        "filters": filters,
        "last_prob": round(filters.get("prob_e11"), 4) if filters.get("prob_e11") is not None else (round(prob, 4) if prob is not None else None),
        "kelly_pct": filters.get("pct_kelly"),
        "active_strategy": filters.get("strategy", "NONE"),
        "position": state.get("position", empty_position()),
        "trades": trades[-50:],
        "equity_history": state.get("equity_history", [])[-300:],
        "prob_history": state.get("prob_history", [])[-300:],
        "regime_history": state.get("regime_history", [])[-1440:],
        "candles_1h": state.get("candles_1h", []),
        "daily_pnl": [{"date": k, "pnl": v} for k, v in sorted(daily_pnl.items())],
    }"""
text = text.replace(old_publish, new_publish)

text = text.replace('state["position"] = {"open": False}', 'state["position"] = empty_position()')

old_run_download = """    # ── Descargar datos de mercado ────────────────────────────────────────────
    print("[E11] Descargando datos 1h BTC y ETH desde yfinance para alinear señales con histórico...")
    df_btc = download_hourly_yfinance("BTC-USD")
    tmp_eth_yf = download_hourly_yfinance("ETH-USD")
    df_eth = tmp_eth_yf[["close"]] if not tmp_eth_yf.empty else pd.DataFrame()

    if df_btc.empty and client:
        print("[E11] yfinance fallo en BTC, usando Bitget como fallback")
        df_btc = get_bitget_candles(client, "BTCUSDT", "1H", limit=1200)
    if df_eth.empty and client:
        print("[E11] yfinance fallo en ETH, usando Bitget como fallback")
        tmp_eth = get_bitget_candles(client, "ETHUSDT", "1H", limit=1200)
        df_eth = tmp_eth[["close"]] if not tmp_eth.empty else pd.DataFrame()

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
            "volume": round(float(row_c.get("vol", 0.0)), 4),
        })
    state["candles_1h"] = candles_snapshot
"""
new_run_download = """    # ── Descargar datos de mercado ────────────────────────────────────────────
    print("[E11] Descargando BTC y ETH por yfinance")
    df_btc, df_btc_1m, btc_live_built = build_live_market_data("BTC-USD")
    df_eth_full, df_eth_1m, eth_live_built = build_live_market_data("ETH-USD")
    df_eth = df_eth_full[["close"]] if df_eth_full is not None and not df_eth_full.empty else pd.DataFrame()

    if (df_btc is None or df_btc.empty) and client:
        print("[E11] yfinance fallo en BTC, usando Bitget 1H como fallback")
        df_btc = get_bitget_candles(client, "BTCUSDT", "1H", limit=1200)

    if (df_eth is None or df_eth.empty) and client:
        print("[E11] yfinance fallo en ETH, usando Bitget 1H como fallback")
        tmp_eth = get_bitget_candles(client, "ETHUSDT", "1H", limit=1200)
        df_eth = tmp_eth[["close"]] if not tmp_eth.empty else pd.DataFrame()

    if df_btc is None or df_btc.empty:
        print("[E11] Sin datos BTC — abortando")
        state["consecutive_errors"] = state.get("consecutive_errors", 0) + 1
        check_auto_pause(state, equity)
        save_state(state)
        return

    state["consecutive_errors"] = 0
    btc_price = float(df_btc["close"].iloc[-1])
    last_candle_ts = df_btc.index[-1].isoformat()
    current_hour_ts = df_btc.index[-1]
    print(
        f"[E11] BTC velas={len(df_btc)} close={btc_price:.2f} "
        f"vela_eval={last_candle_ts} live_1m={'SI' if btc_live_built else 'NO'}"
    )

    candles_snapshot = []
    for ts_c, row_c in df_btc.tail(240).iterrows():
        candles_snapshot.append({
            "ts": ts_c.isoformat(),
            "open": round(float(row_c["open"]), 2),
            "high": round(float(row_c["high"]), 2),
            "low": round(float(row_c["low"]), 2),
            "close": round(float(row_c["close"]), 2),
            "volume": round(float(row_c.get("vol", 0.0)), 4),
        })
    state["candles_1h"] = candles_snapshot
"""
text = text.replace(old_run_download, new_run_download)

old_filters = """    vol_ok    = vol_pct <= GARCH_VOL_UMBRAL
    ema_ok    = ema_slope > 0

    filters["pipeline_mode"] = pipeline_mode
    filters["regime"]        = regime
    filters["vol_percentile"]= round(vol_pct, 4)
    filters["vol_ok"]        = vol_ok
    filters["ema200_slope"]  = round(ema_slope, 6)
    filters["ema200_ok"]     = ema_ok

    print(f"[BOT] Pipeline: {pipeline_mode} | Regimen: {regime} | Vol: {vol_pct:.3f} | EMA200 slope: {ema_slope:.6f}")

    # ── Seleccion de submodelo segun regimen HMM (igual que pipeline TFG) ──────
    # BULL -> xgb_bull.pkl    |    BEAR/SIDEWAYS -> xgb_bear.pkl
    if regime == "BULL":
        model_active = load_model("models/xgb_bull.pkl")
        model_tag = "xgb_bull"
    else:
        model_active = load_model("models/xgb_bear.pkl")
        model_tag = "xgb_bear"
"""
new_filters = """    vol_ok = vol_pct <= GARCH_VOL_UMBRAL
    ema_ok = ema_slope > 0
    regime_e11_ok = regime == "BULL"

    filters["pipeline_mode"] = pipeline_mode
    filters["regime"] = regime
    filters["vol_percentile"] = round(vol_pct, 4)
    filters["vol_ok"] = vol_ok
    filters["ema200_slope"] = round(ema_slope, 6)
    filters["ema200_ok"] = ema_ok
    filters["regime_e11_ok"] = regime_e11_ok
    filters["daily_context_ts"] = str(current_hour_ts.date())

    print(f"[BOT] Pipeline={pipeline_mode} | Regimen={regime} | Vol={vol_pct:.3f} | EMA200 slope={ema_slope:.6f}")

    # ── Seleccion de submodelo segun regimen HMM (igual que pipeline TFG) ──────
    if regime == "BULL":
        model_active = load_model("models/xgb_bull.pkl")
        model_tag = "xgb_bull"
    else:
        model_active = load_model("models/xgb_bear.pkl")
        model_tag = "xgb_bear"
    filters["submodel"] = model_tag
"""
text = text.replace(old_filters, new_filters)

old_e11 = """    print("[BOT] Intentando E11 (macro + Kelly + shorts off)...")
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
"""
new_e11 = """    print("[BOT] Intentando E11")
    in_dead_zone_e11 = abs(xgb_prob - 0.5) <= DELTA
    filters["in_dead_zone_e11"] = in_dead_zone_e11
    filters["in_dead_zone_e6"] = None

    if not vol_ok:
        print(f"[E11] Bloqueado por volatilidad: {vol_pct:.3f} > {GARCH_VOL_UMBRAL}")
    elif in_dead_zone_global:
        print(f"[E11] Bloqueado por dead zone global")
    elif not regime_e11_ok:
        print(f"[E11] Bloqueado por regimen: {regime}")
    elif in_dead_zone_e11:
        print(f"[E11] Bloqueado por dead zone E11")
    elif xgb_prob <= 0.5:
        print(f"[E11] Bloqueado porque solo LONG y prob={xgb_prob:.4f}")
    elif not ema_ok:
        print(f"[E11] Bloqueado por EMA200 slope={ema_slope:.6f}")
    else:
        signal = "LONG"
        strategy = "E11"
        prob = xgb_prob
        pct_e11 = kelly_sizing(prob)
        filters["pct_kelly"] = round(pct_e11, 4)
        print(f"[E11] *** SENAL LONG valida | prob={xgb_prob:.4f} | pct={pct_e11:.4f} ***")
"""
text = text.replace(old_e11, new_e11)

old_e6 = """    if signal == "HOLD" and not state.get("position", {}).get("open"):
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
"""
new_e6 = """    if signal == "HOLD" and not state.get("position", {}).get("open"):
        print("[BOT] E11 no entra, intentando E6")
        in_dead_zone_e6 = abs(xgb_prob - 0.5) <= E6_DELTA
        filters["in_dead_zone_e6"] = in_dead_zone_e6

        if not vol_ok:
            print(f"[E6] Bloqueado por volatilidad: {vol_pct:.3f} > {GARCH_VOL_UMBRAL}")
        elif in_dead_zone_global:
            print(f"[E6] Bloqueado por dead zone global")
        elif in_dead_zone_e6:
            print(f"[E6] Bloqueado por dead zone E6")
        else:
            signal = "LONG" if xgb_prob > 0.5 else "SHORT"
            strategy = "E6"
            prob = xgb_prob
            filters["pct_e6"] = round(E6_PCT, 4)
            print(f"[E6] *** SENAL {signal} valida | prob={xgb_prob:.4f} ***")
"""
text = text.replace(old_e6, new_e6)

path.write_text(text, encoding="utf-8")
print("bot.py actualizado")
