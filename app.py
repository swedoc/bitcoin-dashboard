from flask import Flask, render_template, jsonify
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from collections import deque

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import request

def _get_conn():
    dsn = os.environ["DATABASE_URL"]
    return psycopg2.connect(dsn, sslmode="require")

def increment_and_get_total():
    ua = (request.headers.get("User-Agent") or "").lower()
    if any(x in ua for x in ["bot", "spider", "crawler", "preview", "monitor"]):
        with _get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT total
                FROM page_views
                ORDER BY id DESC
                LIMIT 1
            """)
            row = cur.fetchone()
        return row["total"] if row else 0
    else:
        with _get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO page_views (viewed_at)
                VALUES (NOW())
                RETURNING id
            """)
            cur.execute("""
                SELECT COUNT(*) AS total
                FROM page_views
            """)
            row = cur.fetchone()
        return row["total"] if row else 0

app = Flask(__name__)

def get_binance_klines(symbol="BTCUSDT", interval="1d", limit=365):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data

def fetch_btc_data():
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        params = {"symbol": "BTCUSDT"}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        price = float(data["lastPrice"])
        high = float(data["highPrice"])
        low = float(data["lowPrice"])
        pct_change = float(data["priceChangePercent"])
        volume = float(data["volume"])

        return {
            "price": price,
            "high": high,
            "low": low,
            "pct_change": pct_change,
            "volume": volume,
            "success": True,
        }
    except Exception as e:
        print("BTC data error:", e)
        return {"success": False}

def fetch_btc_fear_greed():
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "data" in data and len(data["data"]) > 0:
            value = int(data["data"][0]["value"])
            classification = data["data"][0]["value_classification"]
            return {"value": value, "classification": classification, "success": True}
    except Exception as e:
        print("Fear & Greed error:", e)
    return {"success": False}

def fetch_btc_dominance():
    try:
        url = "https://api.coingecko.com/api/v3/global"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        dominance = data["data"]["market_cap_percentage"]["btc"]
        return {"dominance": dominance, "success": True}
    except Exception as e:
        print("Dominance error:", e)
        return {"success": False}

def fetch_onchain_data():
    onchain = {}
    try:
        url_hashrate = "https://api.blockchain.info/q/hashrate"
        r1 = requests.get(url_hashrate, timeout=10)
        r1.raise_for_status()
        hashrate = float(r1.text) / 1e9  # GH/s -> TH/s? Beroende på enhet.
        onchain["hashrate"] = hashrate
    except Exception as e:
        print("Hashrate error:", e)

    try:
        url_difficulty = "https://api.blockchain.info/q/getdifficulty"
        r2 = requests.get(url_difficulty, timeout=10)
        r2.raise_for_status()
        difficulty = float(r2.text)
        onchain["difficulty"] = difficulty
    except Exception as e:
        print("Difficulty error:", e)

    try:
        url_mempool = "https://mempool.space/api/mempool"
        r3 = requests.get(url_mempool, timeout=10)
        r3.raise_for_status()
        m = r3.json()
        onchain["mempool_tx"] = m.get("count", None)
    except Exception as e:
        print("Mempool error:", e)

    try:
        url_blocks = "https://mempool.space/api/blocks/tip/height"
        r4 = requests.get(url_blocks, timeout=10)
        r4.raise_for_status()
        height = int(r4.text)
        onchain["block_height"] = height

        next_halving_height = 840000
        blocks_to_halving = next_halving_height - height
        avg_block_time = 10
        minutes_to_halving = blocks_to_halving * avg_block_time
        days_to_halving = minutes_to_halving / 60 / 24
        onchain["days_to_halving"] = days_to_halving
    except Exception as e:
        print("Block height / halving error:", e)

    return onchain

def calc_long_term_trend():
    try:
        klines = get_binance_klines(limit=365)
        closes = [float(k[4]) for k in klines]
        df = pd.DataFrame(closes, columns=["close"])

        df["ma50"] = df["close"].rolling(window=50).mean()
        df["ma200"] = df["close"].rolling(window=200).mean()

        latest_close = df["close"].iloc[-1]
        latest_ma50 = df["ma50"].iloc[-1]
        latest_ma200 = df["ma200"].iloc[-1]

        bull_50_200 = latest_ma50 > latest_ma200
        price_vs_200 = latest_close > latest_ma200

        return {
            "latest_close": latest_close,
            "ma50": latest_ma50,
            "ma200": latest_ma200,
            "bull_50_200": bull_50_200,
            "price_vs_200": price_vs_200,
            "success": True,
        }
    except Exception as e:
        print("Long-term trend error:", e)
        return {"success": False}

def calc_halving_cyclicity():
    try:
        url = "https://api.blockchain.info/q/getblockcount"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        height = int(r.text)

        halving_interval = 210000
        cycle_number = height // halving_interval + 1
        blocks_in_cycle = height % halving_interval
        progress = blocks_in_cycle / halving_interval

        return {
            "cycle_number": cycle_number,
            "cycle_progress": progress,
            "success": True,
        }
    except Exception as e:
        print("Halving cyclicity error:", e)
        return {"success": False}

def fetch_funding_rate():
    try:
        url = "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        funding_rate = float(data["lastFundingRate"]) * 100
        return {"funding_rate": funding_rate, "success": True}
    except Exception as e:
        print("Funding rate error:", e)
        return {"success": False}

def fetch_binance_open_interest():
    try:
        url = "https://fapi.binance.com/futures/data/openInterestHist"
        params = {"symbol": "BTCUSDT", "period": "5m", "limit": 1}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if len(data) > 0:
            oi = float(data[0]["sumOpenInterestValue"]) / 1e9
            return {"open_interest": oi, "success": True}
    except Exception as e:
        print("Open interest error:", e)
        return {"success": False}

def fetch_liquidity_pools():
    try:
        return {
            "liquidity_up": 250,
            "liquidity_down": 180,
            "success": True
        }
    except Exception as e:
        print("Liquidity pool error:", e)
        return {"success": False}

PGL_WINDOW = 30
pgl_deque = deque(maxlen=PGL_WINDOW)

def smma(series, length):
    if length <= 0:
        return [None] * len(series)
    n = len(series)
    out = [None] * n
    if n < length:
        return out
    sma = sum(series[:length]) / length
    out[length - 1] = sma
    for i in range(length, n):
        out[i] = (out[i - 1] * (length - 1) + series[i]) / length
    return out

def calc_pgl_status():
    """
    Beräknar PGL-status med samma logik som i /api/pgl och ditt TradingView-skript:
    - v1, m1, m2, v2 är SMMA på HL2 = (high + low) / 2
    - p2 = neutralflagga (mellanläge) när relationerna är blandade
    - annars:
        b = (v1 < v2)  -> Bear (nedåtfas)
        not b          -> Bull (uppåtfas)
    Returnerar: (status_text, färgnyckel), score (-1, 0, 1)
    """
    klines = get_binance_klines(limit=60)
    # sista raden på Binance är pågående D1-bar, använd bara stängda barer
    klines = klines[:-1]

    if not klines or len(klines) < 30:
        return ("Neutral", "orange"), 0

    # HL2 = (high + low) / 2, samma som i /api_pgl
    hl2 = []
    for k in klines:
        try:
            high = float(k[2])
            low = float(k[3])
            hl2.append((high + low) / 2.0)
        except Exception:
            return ("Neutral", "orange"), 0

    v1 = smma(hl2, 15)
    m1 = smma(hl2, 19)
    m2 = smma(hl2, 25)
    v2 = smma(hl2, 29)

    v1_last, m1_last, m2_last, v2_last = v1[-1], m1[-1], m2[-1], v2[-1]
    if None in (v1_last, m1_last, m2_last, v2_last):
        return ("Neutral", "orange"), 0

    # Liten tolerans: "nästan lika" ska bli Neutral
    eps = 1e-5
    a = (v1_last + eps) < m1_last      # v1 < m1
    b = (v1_last + eps) < v2_last      # v1 < v2
    c = (m2_last + eps) < v2_last      # m2 < v2

    # p2 = mellanläge (vit i TradingView) när relationerna är blandade
    p2 = (a != b) or (c != b)

    if (not p2) and b:
        # Nedåtfas (violett i din PGL-logik)
        return ("Bear", "red-dark"), -1
    elif (not p2) and (not b):
        # Uppåtfas (orange i din PGL-logik)
        return ("Bull", "green-dark"), 1
    else:
        # Mellanläge (vit i din PGL-logik)
        return ("Neutral", "orange"), 0

def calculate_atr(lookback=250, atr_period=14):
    klines = get_binance_klines(limit=lookback + atr_period)
    highs, lows, closes = [], [], []
    for k in klines:
        try:
            highs.append(float(k[2])); lows.append(float(k[3])); closes.append(float(k[4]))
        except:
            return None, []
    if len(closes) < atr_period + 1:
        return None, []
    tr_list = []
    for i in range(1, len(closes)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i - 1]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_list.append(tr)
    atr_series = pd.Series(tr_list).rolling(window=atr_period).mean()
    latest_atr = atr_series.iloc[-1]
    return latest_atr, atr_series.tolist()

def calc_volatility_regime():
    try:
        latest_atr, atr_series = calculate_atr()
        if latest_atr is None:
            return {"success": False}

        atr_array = np.array(atr_series)
        low_threshold = np.percentile(atr_array[~np.isnan(atr_array)], 30)
        high_threshold = np.percentile(atr_array[~np.isnan(atr_array)], 70)

        if latest_atr < low_threshold:
            regime = "Låg volatilitet"
        elif latest_atr > high_threshold:
            regime = "Hög volatilitet"
        else:
            regime = "Medelhög volatilitet"

        return {
            "latest_atr": latest_atr,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "regime": regime,
            "success": True,
        }
    except Exception as e:
        print("Volatility regime error:", e)
        return {"success": False}

def fetch_onchain_glassnode_dummy():
    try:
        return {
            "sopr": 0.98,
            "nrpl": 0.03,
            "hoder_net_pos": 1500,
            "success": True,
        }
    except Exception as e:
        print("On-chain dummy error:", e)
        return {"success": False}

def fetch_macro_dummy():
    try:
        return {
            "us_10y": 4.3,
            "dxy": 102.5,
            "spx": 5200,
            "success": True,
        }
    except Exception as e:
        print("Macro dummy error:", e)
        return {"success": False}

def enrich_pgl_with_context(pgl_score, funding_rate, dominance, volatility_regime, fear_greed):
    try:
        context = {}
        safe_dominance = dominance.get("dominance") if dominance.get("success") else None
        safe_fng_value = fear_greed.get("value") if fear_greed.get("success") else None
        safe_fng_class = fear_greed.get("classification") if fear_greed.get("success") else None
        safe_vol_regime = volatility_regime.get("regime") if volatility_regime.get("success") else None
        safe_funding = funding_rate.get("funding_rate") if funding_rate.get("success") else None

        aggressiveness = 0

        if pgl_score == 1:
            aggressiveness += 1
        elif pgl_score == -1:
            aggressiveness -= 1

        if safe_funding is not None:
            if safe_funding > 0.05:
                aggressiveness += 1
                context["funding_bias"] = "Longs dominerar"
            elif safe_funding < -0.05:
                aggressiveness -= 1
                context["funding_bias"] = "Shorts dominerar"
            else:
                context["funding_bias"] = "Neutral funding"

        if safe_dominance is not None:
            if safe_dominance > 50:
                aggressiveness += 1
                context["dominance_bias"] = "BTC stark"
            elif safe_dominance < 40:
                aggressiveness -= 1
                context["dominance_bias"] = "Altcoinklimat"
            else:
                context["dominance_bias"] = "Neutral dominans"

        if safe_fng_value is not None:
            if safe_fng_value < 25:
                aggressiveness += 1
                context["fng_bias"] = "Extrem rädsla (kontraindikator, möjligt läge)"
            elif safe_fng_value > 75:
                aggressiveness -= 1
                context["fng_bias"] = "Extrem girighet (risk för topp)"
            else:
                context["fng_bias"] = f"Sentiment: {safe_fng_class}"

        if safe_vol_regime is not None:
            if "Låg" in safe_vol_regime:
                aggressiveness += 0.5
                context["volatility_bias"] = "Låg volatilitet (kan föregå större rörelse)"
            elif "Hög" in safe_vol_regime:
                aggressiveness -= 0.5
                context["volatility_bias"] = "Hög volatilitet (risk för slagig marknad)"
            else:
                context["volatility_bias"] = "Normal volatilitet"

        if aggressiveness >= 2:
            mode = "Aggressiv uppåt-bias"
        elif aggressiveness >= 1:
            mode = "Måttlig uppåt-bias"
        elif aggressiveness <= -2:
            mode = "Aggressiv nedåt-bias"
        elif aggressiveness <= -1:
            mode = "Måttlig nedåt-bias"
        else:
            mode = "Neutral till försiktig"

        context["aggregated_mode"] = mode
        context["aggregated_score"] = aggressiveness
        return context
    except Exception as e:
        print("Context aggregation error:", e)
        return {"aggregated_mode": "Okänt", "aggregated_score": 0}

@app.route("/")
def index():
    try:
        view_count = increment_and_get_total()
    except Exception as e:
        print("View counter error:", e)
        view_count = 0

    btc_data = fetch_btc_data()
    onchain = fetch_onchain_data()
    lt_trend = calc_long_term_trend()
    halving_cycle = calc_halving_cyclicity()
    fear_greed = fetch_btc_fear_greed()
    dominance = fetch_btc_dominance()
    funding = fetch_funding_rate()
    oi = fetch_binance_open_interest()
    liquidity = fetch_liquidity_pools()
    vol_regime = calc_volatility_regime()
    glassnode_dummy = fetch_onchain_glassnode_dummy()
    macro_dummy = fetch_macro_dummy()

    (pgl_label, pgl_color), pgl_score = calc_pgl_status()
    pgl_deque.append(pgl_score)
    agg_context = enrich_pgl_with_context(
        pgl_score, funding, dominance, vol_regime, fear_greed
    )

    now_utc = datetime.now(pytz.utc)
    stockholm_tz = pytz.timezone("Europe/Stockholm")
    now_stockholm = now_utc.astimezone(stockholm_tz)
    last_updated = now_stockholm.strftime("%Y-%m-%d %H:%M:%S")

    return render_template(
        "index.html",
        btc_data=btc_data,
        onchain=onchain,
        lt_trend=lt_trend,
        halving_cycle=halving_cycle,
        fear_greed=fear_greed,
        dominance=dominance,
        funding=funding,
        open_interest=oi,
        liquidity=liquidity,
        volatility=vol_regime,
        glassnode=glassnode_dummy,
        macro=macro_dummy,
        pgl_label=pgl_label,
        pgl_color=pgl_color,
        pgl_score=pgl_score,
        pgl_history=list(pgl_deque),
        agg_context=agg_context,
        last_updated=last_updated,
        total_views=view_count,
    )

@app.route("/api/raw")
def api_raw():
    btc_data = fetch_btc_data()
    onchain = fetch_onchain_data()
    lt_trend = calc_long_term_trend()
    halving_cycle = calc_halving_cyclicity()
    fear_greed = fetch_btc_fear_greed()
    dominance = fetch_btc_dominance()
    funding = fetch_funding_rate()
    oi = fetch_binance_open_interest()
    liquidity = fetch_liquidity_pools()
    vol_regime = calc_volatility_regime()
    glassnode_dummy = fetch_onchain_glassnode_dummy()
    macro_dummy = fetch_macro_dummy()
    (pgl_label, pgl_color), pgl_score = calc_pgl_status()
    agg_context = enrich_pgl_with_context(
        pgl_score, funding, dominance, vol_regime, fear_greed
    )

    return jsonify({
        "btc_data": btc_data,
        "onchain": onchain,
        "lt_trend": lt_trend,
        "halving_cycle": halving_cycle,
        "fear_greed": fear_greed,
        "dominance": dominance,
        "funding": funding,
        "open_interest": oi,
        "liquidity": liquidity,
        "volatility": vol_regime,
        "glassnode": glassnode_dummy,
        "macro": macro_dummy,
        "pgl": {
            "label": pgl_label,
            "color": pgl_color,
            "score": pgl_score,
            "history": list(pgl_deque),
        },
        "aggregated_context": agg_context,
    })

@app.route("/api/pgl")
def api_pgl():
    klines = get_binance_klines(limit=60)
    klines = klines[:-1]

    hl2 = []
    for k in klines:
        high = float(k[2]); low = float(k[3])
        hl2.append((high + low) / 2.0)

    v1 = smma(hl2, 15); m1 = smma(hl2, 19); m2 = smma(hl2, 25); v2 = smma(hl2, 29)
    v1_last, m1_last, m2_last, v2_last = v1[-1], m1[-1], m2[-1], v2[-1]

    eps = 1e-5
    a = (v1_last + eps) < m1_last
    b = (v1_last + eps) < v2_last
    c = (m2_last + eps) < v2_last
    p2 = (a != b) or (c != b)

    if (not p2) and b:
        status, color, score = "Bear", "red-dark", -1
    elif (not p2) and (not b):
        status, color, score = "Bull", "green-dark", 1
    else:
        status, color, score = "Neutral", "orange", 0

    return jsonify({
        "status": status,
        "color": color,
        "score": score,
        "v1_last": v1_last,
        "m1_last": m1_last,
        "m2_last": m2_last,
        "v2_last": v2_last,
        "a_v1_lt_m1": a,
        "b_v1_lt_v2": b,
        "c_m2_lt_v2": c,
        "p2_neutral_flag": p2
    })

if __name__ == "__main__":
    app.run(debug=True)
