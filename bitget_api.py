"""
bitget_api.py
=============
Wrapper Bitget USDT Futures API v2.
Autenticacion: HMAC-SHA256 sobre (timestamp + method + path + body).
Solo requests + hmac estandar, sin librerias de exchange externas.
"""

import base64
import hashlib
import hmac
import json
import math
import os
import time
from datetime import datetime, timezone

import requests

BASE_URL = "https://api.bitget.com"
PRODUCT  = "USDT-FUTURES"
SYMBOL   = "BTCUSDT"


class BitgetClient:
    def __init__(self, api_key, secret, passphrase):
        self.api_key    = api_key
        self.secret     = secret
        self.passphrase = passphrase
        self.session    = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "locale": "en-US",
        })

    def _sign_b64(self, timestamp, method, path, body=""):
        msg = timestamp + method.upper() + path + (body or "")
        raw = hmac.new(
            self.secret.encode("utf-8"),
            msg.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(raw).decode()

    def _headers(self, method, path, body=""):
        ts  = str(int(time.time() * 1000))
        sig = self._sign_b64(ts, method, path, body)
        return {
            "ACCESS-KEY":        self.api_key,
            "ACCESS-SIGN":       sig,
            "ACCESS-TIMESTAMP":  ts,
            "ACCESS-PASSPHRASE": self.passphrase,
        }

    def _get(self, path, params=None):
        qs = ""
        if params:
            qs = "?" + "&".join(f"{k}={v}" for k, v in params.items())
        full_path = path + qs
        headers   = self._headers("GET", full_path)
        resp      = self.session.get(BASE_URL + full_path, headers=headers, timeout=10)
        data = resp.json()
        if data.get("code") != "00000":
            raise RuntimeError(f"Bitget GET {path} error: {data}")
        return data

    def _post(self, path, body_dict):
        body_str = json.dumps(body_dict, separators=(",", ":"))
        headers  = self._headers("POST", path, body_str)
        resp     = self.session.post(BASE_URL + path, headers=headers,
                                     data=body_str, timeout=10)
        data = resp.json()
        if data.get("code") != "00000":
            raise RuntimeError(f"Bitget POST {path} error: {data}")
        return data

    def get_price(self, symbol=SYMBOL):
        data   = self._get("/api/v2/mix/market/ticker",
                           {"symbol": symbol, "productType": PRODUCT})
        ticker = data["data"]
        if isinstance(ticker, list):
            ticker = ticker[0]
        return float(ticker["lastPr"])

    def get_contract_info(self, symbol=SYMBOL):
        data      = self._get("/api/v2/mix/market/contracts",
                              {"symbol": symbol, "productType": PRODUCT})
        contracts = data["data"]
        if isinstance(contracts, list):
            return contracts[0]
        return contracts

    def get_step_size(self, symbol=SYMBOL):
        return float(self.get_contract_info(symbol).get("sizeMultiplier", 0.001))

    def get_min_size(self, symbol=SYMBOL):
        return float(self.get_contract_info(symbol).get("minTradeNum", 0.001))

    def round_qty(self, qty, step):
        precision = max(0, -int(math.floor(math.log10(step))))
        return round(math.floor(qty / step) * step, precision)

    def get_balance(self):
        data = self._get("/api/v2/mix/account/account",
                         {"symbol": SYMBOL, "productType": PRODUCT, "marginCoin": "USDT"})
        acc  = data["data"]
        return {
            "available":     float(acc.get("available", 0)),
            "equity":        float(acc.get("usdtEquity", acc.get("equity", 0))),
            "unrealized_pl": float(acc.get("unrealizedPL", 0)),
        }

    def get_positions(self, symbol=SYMBOL):
        data      = self._get("/api/v2/mix/position/single-position",
                              {"symbol": symbol, "productType": PRODUCT, "marginCoin": "USDT"})
        positions = data.get("data", [])
        return [p for p in positions if float(p.get("total", 0)) > 0]

    def has_open_position(self, symbol=SYMBOL):
        return len(self.get_positions(symbol)) > 0

    def set_leverage(self, symbol, leverage, margin_mode="isolated"):
        return self._post("/api/v2/mix/account/set-leverage", {
            "symbol":      symbol,
            "productType": PRODUCT,
            "marginCoin":  "USDT",
            "leverage":    str(int(leverage)),
            "holdSide":    "long_short",
        })

    def set_margin_mode(self, symbol, margin_mode="isolated"):
        try:
            return self._post("/api/v2/mix/account/set-margin-mode", {
                "symbol":      symbol,
                "productType": PRODUCT,
                "marginCoin":  "USDT",
                "marginMode":  margin_mode,
            })
        except RuntimeError:
            return {}

    def place_order(self, symbol, direction, size_usdt, leverage):
        """
        Abre orden de mercado SIN TP/SL.
        El cierre se gestiona desde bot.py al cambiar de vela 1h.
        size_usdt: notional total (margen * leverage ya aplicado).
        """
        mkt_px = self.get_price(symbol)
        step   = self.get_step_size(symbol)
        min_sz = self.get_min_size(symbol)
        qty    = self.round_qty(size_usdt / mkt_px, step)

        # Abortar si la qty calculada es menor que el minimo del exchange
        if qty < min_sz:
            raise ValueError(
                f"Qty calculada {qty} BTC < minimo del exchange {min_sz} BTC. "
                f"size_usdt={size_usdt:.2f} USDT, precio={mkt_px:.0f}. "
                f"Capital insuficiente para esta operativa."
            )

        if qty <= 0:
            raise ValueError(f"Qty={qty} invalida. size_usdt={size_usdt}")

        self.set_margin_mode(symbol, "isolated")
        self.set_leverage(symbol, leverage, "isolated")

        ts_str    = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        client_id = f"e11_{direction[0].upper()}_{ts_str}"

        body = {
            "symbol":      symbol,
            "productType": PRODUCT,
            "marginMode":  "isolated",
            "marginCoin":  "USDT",
            "size":        str(qty),
            "side":        direction,
            "tradeSide":   "open",
            "orderType":   "market",
            "clientOid":   client_id,
        }
        data     = self._post("/api/v2/mix/order/place-order", body)
        order_id = data["data"].get("orderId")
        print(f"  [BITGET] Orden OK: {direction.upper()} {qty} BTC @ ~{mkt_px:.0f}  lev={leverage}x  orderId={order_id}")

        return {"orderId": order_id, "qty": qty, "entry_px": mkt_px}

    def close_position(self, symbol=SYMBOL):
        try:
            data = self._post("/api/v2/mix/order/close-positions", {
                "symbol":      symbol,
                "productType": PRODUCT,
            })
            print(f"  [BITGET] Posicion cerrada para {symbol}")
            return data
        except Exception as e:
            print(f"  [BITGET] Error cerrando posicion: {e}")
            return {}


def client_from_env():
    key = os.environ.get("BITGET_API_KEY",    "")
    sec = os.environ.get("BITGET_API_SECRET", "")
    pp  = os.environ.get("BITGET_PASSPHRASE", "")
    if not all([key, sec, pp]):
        raise EnvironmentError("Faltan: BITGET_API_KEY, BITGET_API_SECRET, BITGET_PASSPHRASE")
    return BitgetClient(key, sec, pp)
