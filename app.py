from flask import Flask, render_template, jsonify
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from collections import deque

# Threshold för funding rate i promille (‰). Ändra här om du vill justera.
FR_THRESHOLD_PER_MILLE = 0.1

app = Flask(__name__)
qfi_history = deque(maxlen=3)

def get_fgi():
    try:
        response = requests.get("https://api.alternative.me/fng/")
        data = response.json()
        return int(data["data"][0]["value"])
    except:
        return None

def get_price(symbol):
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        response = requests.get(url)
        return float(response.json()["price"])
    except:
        return None

def get_funding_rate(symbol):
    try:
        url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
        response = requests.get(url)
        return float(response.json().get("lastFundingRate", None))
    except:
        return None

def get_open_interest(symbol="BTCUSDT"):
    try:
        url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
        response = requests.get(url)
        data = response.json()
        price = get_price(symbol)
        if price is None:
            return None
        val = float(data["openInterest"]) * price / 1e9
        return val
    except:
        return None

def get_volume(symbol="BTCUSDT"):
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/24hr?symbol={symbol}"
        response = requests.get(url)
        data = response.json()
        val = float(data["quoteVolume"]) / 1e9
        return val
    except:
        return None

def get_cet_time():
    cet = pytz.timezone("Europe/Stockholm")
    now = datetime.now(cet)
    return now.strftime("%Y-%m-%d %H:%M:%S")

def get_binance_klines(symbol="BTCUSDT", interval="1d", limit=500):
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        response = requests.get(url, params=params, timeout=10)
        return response.json()
    except:
        return []

def smma(series, length):
    smma_values = []
    for i in range(len(series)):
        if i < length:
            smma_values.append(sum(series[:i+1])/(i+1))
        else:
            prev = smma_values[-1]
            smma_values.append((prev*(length-1) + series[i]) / length)
    return smma_values

def calc_pgl_status():
    klines = get_binance_klines(limit=60)
    if not klines or len(klines) < 30:
        return ("Neutral", "orange"), 0
    hl2 = []
    for k in klines:
        try:
            high = float(k[2]); low = float(k[3])
            hl2.append((high + low)/2)
        except:
            return ("Neutral", "orange"), 0
    v1 = smma(hl2, 15)
    m1 = smma(hl2, 19)
    m2 = smma(hl2, 25)
    v2 = smma(hl2, 29)
    try:
        p2 = ((v1[-1]<m1[-1]) != (v1[-1]<v2[-1])) or ((m2[-1]<v2[-1]) != (v1[-1]<v2[-1]))
        p3 = (not p2) and (v1[-1]<v2[-1])
        p1 = (not p2) and (not p3)
    except:
        return ("Neutral", "orange"), 0
    if p1:
        return ("Bull", "green-dark"), 1
    elif p2:
        return ("Neutral", "orange"), 0
    else:
        return ("Bear", "red-dark"), -1

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
        high = highs[i]; low = lows[i]; prev_close = closes[i-1]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_list.append(tr)
    atr_series = pd.Series(tr_list).rolling(window=atr_period).mean().dropna().tolist()
    if not atr_series:
        return None, []
    latest_atr = atr_series[-1]
    ratios = []
    price_slice = closes[1:]
    for i, atr in enumerate(atr_series):
        idx = i + (len(price_slice) - len(atr_series))
        if 0 <= idx < len(price_slice):
            price = price_slice[idx]
            if price and atr:
                ratios.append(float(atr)/float(price))
    return float(latest_atr), ratios

def determine_dynamic_atr_ratio(symbol="BTCUSDT", lookback=250, atr_period=14, pct=0.20):
    latest_atr, ratios = calculate_atr(lookback=lookback, atr_period=atr_period)
    if not ratios:
        return None
    try:
        return float(np.percentile(ratios, pct*100))
    except:
        return None

def score(param, value):
    if value is None:
        return 0
    if param == "fgi":
        if value <= 24: return -3
        elif value <= 49: return -1
        elif value <= 54: return 0
        elif value <= 74: return 1
        else: return 3
    elif param == "fr":
         # Bearish < -FR_THRESHOLD_PER_MILLE; Neutral mellan ±; Bullish >
         if value < -FR_THRESHOLD_PER_MILLE:      
             return -1
         elif value <= FR_THRESHOLD_PER_MILLE:  
             return 0
         else:              
             return 1
    elif param == "oi":
        if value < 6: return -1
        elif value <= 10: return 0
        else: return 1
    elif param == "vol":
        if value < 10: return -1
        elif value <= 20: return 0
        else: return 1
    return 0

def interpret_pfi(total_score):
    if total_score <= -3:
        return "Bearish", "red", "Recommendation: SELL"
    elif -2 <= total_score <= 2:
        return "Neutral", "orange", "Recommendation: HOLD"
    else:
        return "Bullish", "green", "Recommendation: BUY"

@app.route("/")
def index():
    fgi = get_fgi()
    price = get_price("BTCUSDT")
    funding_rate = get_funding_rate("BTCUSDT")
    # Steg 1: Konvertera funding_rate till promille för logiken
    fr_perm = funding_rate * 1000 if isinstance(funding_rate, (int, float)) else None
    # Formatera funding rate för visning i promille
    funding_rate_str = f"{fr_perm:.2f} ‰" if fr_perm is not None else "N/A"

    open_interest = get_open_interest()
    volume = get_volume()
    timestamp = get_cet_time()

    dynamic_ratio = determine_dynamic_atr_ratio(symbol="BTCUSDT", lookback=250, atr_period=14, pct=0.20)
    if dynamic_ratio is None:
        dynamic_ratio = 0.01
    atr = None; atr_threshold = None; volatile = True
    if isinstance(price, (int, float)):
        latest_atr, _ = calculate_atr(lookback=250, atr_period=14)
        if latest_atr is not None:
            atr = latest_atr
            atr_threshold = dynamic_ratio * price
            volatile = (atr >= atr_threshold)

    klines = get_binance_klines(limit=21)
    if len(klines) >= 21:
        try:
            closes = [float(k[4]) for k in klines]
            sma_20 = sum(closes[:-1]) / 20
            price_dir = 1 if closes[-1] > sma_20 else -1
        except:
            price_dir = 0
    else:
        price_dir = 0

    pgl_status, pgl_score = calc_pgl_status()

    fgi_score = score("fgi", fgi)
    fr_score = score("fr", fr_perm)
    if volume is None:
        vol_score = 0
    else:
        if volatile:
            if volume > 10:
                vol_score = 1 * price_dir
            elif volume < 5:
                vol_score = -1
            else:
                vol_score = 0
        else:
            vol_score = 0

    if open_interest is None:
        oi_score = 0
    else:
        if volatile:
            if open_interest > 10:
                oi_score = 1 * price_dir
            elif open_interest < 6:
                oi_score = -1
            else:
                oi_score = 0
        else:
            oi_score = 0

    pgl_weight = 2.0
    total_score = fgi_score + fr_score + vol_score + oi_score + pgl_score * pgl_weight
    qfi_history.append(total_score)
    smoothed = sum(qfi_history) / len(qfi_history)
    pfi_text, pfi_color, pfi_expl = interpret_pfi(smoothed)
    pfi_value = round(smoothed, 2)

    if smoothed >= 3:
        trade_signal = "Long"
        trade_class = "green"
    elif smoothed <= -3:
        trade_signal = "Short"
        trade_class = "red"
    else:
        trade_signal = "Neutral"
        trade_class = "orange"

    def tolka(param, value, price_dir=None):
        if value is None:
            return "N/A", "gray"
        if param == "fgi":
            if value <= 24: return "Extreme Fear", "red-dark"
            elif value <= 49: return "Fear", "red-light"
            elif value <= 54: return "Neutral", "gray"
            elif value <= 74: return "Greed", "green-light"
            else: return "Extreme Greed", "green-dark"
        if param == "fr":
            if value is None:
                return "N/A", "gray"
            if value < -FR_THRESHOLD_PER_MILLE:
                return "Bearish", "red-dark"
            elif value <= FR_THRESHOLD_PER_MILLE:
                return "Neutral", "orange"
            else:
                return "Bullish", "green-dark"
        if param == "oi":
            if not volatile:
                return "Low volatility → ignored", "gray"
            if value > 10:
                return ("High OI on uptrend – bullish", "green-dark") if price_dir > 0 else ("High OI on downtrend – bearish", "red-dark")
            elif value < 6:
                return "Low OI", "gray"
            else:
                return "Neutral OI", "orange"
        if param == "vol":
            if not volatile:
                return "Low volatility → ignored", "gray"
            if value > 20:
                return ("High volume on uptrend – bullish", "green-dark") if price_dir > 0 else ("High volume on downtrend – bearish", "red-dark")
            elif value < 10:
                return "Low volume", "gray"
            else:
                return "Neutral volume", "orange"
        return "Unknown", "gray"

    price_str = f"{price:,.2f}" if isinstance(price, (int, float)) else "N/A"

    open_interest_str = f"{open_interest:,.2f}" if isinstance(open_interest, (int, float)) else "N/A"
    volume_str = f"{volume:,.2f}" if isinstance(volume, (int, float)) else "N/A"
    # Avrunda till heltal och formatera med tusentalsavgränsare
    if isinstance(atr, (int, float)):
        atr_str = f"{int(round(atr)):,}"
    else:
        atr_str = "N/A"

    if isinstance(atr_threshold, (int, float)):
        atr_threshold_str = f"{int(round(atr_threshold)):,}"
    else:
        atr_threshold_str = "N/A"

    def compare_pair(par, a, b):
        if not volatile:
            return ("Unclear", "orange", "Low volatility → signals ignored")
        if a is None or b is None:
            return ("Unknown", "orange", "Data unavailable")
        if par == "fr+vol":
            if a > 0.01 and b > 10:
                return ("Bullish", "green-dark", "High FR & volume = strong conviction")
            elif a < -0.01 and b > 10:
                return ("Bearish", "red-dark", "Negative FR & high volume = strong short pressure")
            elif abs(a) <= 0.01 and 5 <= b <= 10:
                return ("Neutral", "orange", "Neutral funding & medium volume")
            else:
                return ("Unclear", "orange", "Inconsistent signal")
        if par == "oi+vol":
            if a > 10 and b > 10:
                return ("Bullish", "green-dark", "High OI & volume = strong trend")
            elif a < 6 and b < 5:
                return ("Bearish", "red-dark", "Low OI & low volume = weak market")
            elif 6 <= a <= 10 and 5 <= b <= 10:
                return ("Neutral", "orange", "Neutral OI & volume")
            else:
                return ("Unclear", "orange", "Mixed signals")
        if par == "fgi+vol":
            if a > 54 and b > 10:
                return ("Bullish", "green-dark", "Greed with high volume = momentum")
            elif a < 50 and b < 5:
                return ("Bearish", "red-dark", "Fear with low volume = no buyers")
            elif 50 <= a <= 54 and 5 <= b <= 10:
                return ("Neutral", "orange", "Neutral sentiment & volume")
            else:
                return ("Unclear", "orange", "No clear signal")
        return ("Unclear", "orange", "No logic applied")

    fr_vol = compare_pair("fr+vol", funding_rate, volume)
    oi_vol = compare_pair("oi+vol", open_interest, volume)
    fgi_vol = compare_pair("fgi+vol", fgi, volume)

    return render_template("index.html",
                           fgi=fgi,
                           fgi_text=tolka("fgi", fgi, price_dir),
                           price=price_str,
                           price_value=price,
                           funding_rate_str=funding_rate_str,
                           fr_text=tolka("fr", fr_perm),
                           fr_score=fr_score,
                           open_interest=open_interest_str,
                           oi_text=tolka("oi", open_interest, price_dir),
                           volume=volume_str,
                           vol_text=tolka("vol", volume, price_dir),
                           atr_str=atr_str,
                           atr_threshold_str=atr_threshold_str,
                           atr_threshold_value=atr_threshold,
                           volatile=volatile,
                           pgl_status=pgl_status,
                           pgl_color=pgl_status[1],
                           pfi_text=pfi_text,
                           pfi_color=pfi_color,
                           pfi_expl=pfi_expl,
                           pfi_value=pfi_value,
                           trade_signal=trade_signal,
                           trade_class=trade_class,
                           timestamp=timestamp,
                           qfi_arrow={
                               "Bullish": '<span style="color:#66ff66;">&#9650;</span>',
                               "Neutral": '<span style="color:#ffa726;">&#9650;</span>',
                               "Bearish": '<span style="color:#d32f2f;">&#9660;</span>'
                           }.get(pfi_text, ""),
                           fr_vol=fr_vol,
                           oi_vol=oi_vol,
                           fgi_vol=fgi_vol,
			   fr_threshold=FR_THRESHOLD_PER_MILLE
			)

@app.route("/api/signal")
def api_signal():
    fgi = get_fgi()
    price = get_price("BTCUSDT")
    funding_rate = get_funding_rate("BTCUSDT")
    fr_perm = funding_rate * 1000 if isinstance(funding_rate, (int, float)) else None
    open_interest = get_open_interest()
    volume = get_volume()

    fgi_score = score("fgi", fgi)
    fr_score = score("fr", fr_perm)
    klines = get_binance_klines(limit=21)
    if len(klines) >= 21:
        try:
            closes = [float(k[4]) for k in klines]
            sma_20 = sum(closes[:-1]) / 20
            price_dir = 1 if closes[-1] > sma_20 else -1
        except:
            price_dir = 0
    else:
        price_dir = 0

    if volume is None:
        vol_score = 0
    else:
        if volume > 10:
            vol_score = 1 * price_dir
        elif volume < 5:
            vol_score = -1
        else:
            vol_score = 0

    if open_interest is None:
        oi_score = 0
    else:
        if open_interest > 10:
            oi_score = 1 * price_dir
        elif open_interest < 6:
            oi_score = -1
        else:
            oi_score = 0

    _, pgl_score = calc_pgl_status()
    total_score = fgi_score + fr_score + vol_score + oi_score + pgl_score * 2.0
    qfi_history.append(total_score)
    smoothed = sum(qfi_history)/len(qfi_history)
    if smoothed >= 3:
        sig = "Long"
    elif smoothed <= -3:
        sig = "Short"
    else:
        sig = "Neutral"
    return jsonify({"signal": sig, "score": smoothed})

if __name__ == "__main__":
    app.run(debug=True)
