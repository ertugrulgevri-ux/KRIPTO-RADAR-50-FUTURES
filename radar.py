
import os
import time
import math
import requests
import statistics
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# FUTURE RADAR 50 - GELISMIS SURUM
# Binance USDT-M Futures
# 50 teknik kontrol / 5 zaman dilimi
# Erken akumulasyon ve kirilim analizi
# ============================================================

VERSION = "2.0"

BASE_URLS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
]

TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]

TF_WEIGHTS = {
    "5m": 10,
    "15m": 20,
    "1h": 30,
    "4h": 25,
    "1d": 15,
}

TOP_N = 100
KLINE_LIMIT = 150
MAX_WORKERS = 6

session = requests.Session()
session.headers.update({"User-Agent": "FutureRadar50/2.0"})

# ============================================================
# API ERISIMI - ALTERNATIF ADRES VE YENIDEN DENEME
# ============================================================

def api_get(path, params=None):
    last_error = None

    for attempt in range(3):
        for base in BASE_URLS:
            try:
                response = session.get(
                    base + path,
                    params=params,
                    timeout=15
                )

                if response.status_code in (403, 451):
                    last_error = Exception(
                        f"API erisim hatasi: HTTP {response.status_code}"
                    )
                    continue

                if response.status_code == 429:
                    time.sleep(2 + attempt * 2)
                    last_error = Exception("API rate limit")
                    continue

                response.raise_for_status()
                return response.json()

            except Exception as exc:
                last_error = exc

        time.sleep(1 + attempt)

    raise Exception(f"Binance API erisilemedi: {last_error}")


# ============================================================
# TEMEL MATEMATIK VE TEKNIK GOSTERGELER
# ============================================================

def ema(values, period):
    if len(values) < period:
        return None

    k = 2 / (period + 1)
    result = sum(values[:period]) / period

    for value in values[period:]:
        result = value * k + result * (1 - k)

    return result


def rsi(values, period=14):
    if len(values) < period + 1:
        return 50

    changes = [
        values[i] - values[i - 1]
        for i in range(1, len(values))
    ]

    gains = [max(x, 0) for x in changes]
    losses = [max(-x, 0) for x in changes]

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100

    return 100 - (100 / (1 + avg_gain / avg_loss))


def bollinger(values, period=20):
    if len(values) < period:
        return None

    data = values[-period:]
    middle = statistics.mean(data)
    deviation = statistics.pstdev(data)

    upper = middle + 2 * deviation
    lower = middle - 2 * deviation

    width = (upper - lower) / middle if middle else 0

    return upper, middle, lower, width


def atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return 0

    trs = []

    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        trs.append(tr)

    return sum(trs[-period:]) / period


def obv_values(closes, volumes):
    result = [0]

    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            result.append(result[-1] + volumes[i])
        elif closes[i] < closes[i - 1]:
            result.append(result[-1] - volumes[i])
        else:
            result.append(result[-1])

    return result


def stochastic_rsi(values):
    if len(values) < 20:
        return 0.5

    recent = values[-14:]
    low = min(recent)
    high = max(recent)

    if high == low:
        return 0.5

    return (values[-1] - low) / (high - low)


def safe_pct(a, b):
    if not b:
        return 0
    return ((a - b) / abs(b)) * 100


# ============================================================
# YENI SINYAL 1: SERT BOLLINGER SIKISMASI
# ============================================================

def bollinger_squeeze(closes):
    if len(closes) < 110:
        return 0

    widths = []

    for end in range(20, len(closes) + 1):
        result = bollinger(closes[:end])
        if result:
            widths.append(result[3])

    if len(widths) < 30:
        return 0

    current = widths[-1]
    history = widths[-100:-1]

    if not history:
        return 0

    rank = sum(x >= current for x in history) / len(history)

    if rank <= 0.05:
        return 100
    elif rank <= 0.10:
        return 85
    elif rank <= 0.20:
        return 65
    elif rank <= 0.30:
        return 35

    return 0


# ============================================================
# YENI SINYAL 2: OBV POZITIF UYUMSUZLUK
# ============================================================

def obv_divergence(lows, closes, volumes):
    if len(closes) < 45:
        return 0

    obv = obv_values(closes, volumes)

    old_lows = lows[-40:-20]
    new_lows = lows[-20:]

    old_index = min(
        range(len(old_lows)),
        key=lambda i: old_lows[i]
    ) + len(lows) - 40

    new_index = min(
        range(len(new_lows)),
        key=lambda i: new_lows[i]
    ) + len(lows) - 20

    price_old = lows[old_index]
    price_new = lows[new_index]

    obv_old = obv[old_index]
    obv_new = obv[new_index]

    if price_old == 0 or obv_old == 0:
        return 0

    price_change = safe_pct(price_new, price_old)
    obv_change = safe_pct(obv_new, obv_old)

    if price_change < -0.3 and obv_change > 2:
        return 100

    if obv_change > 5 and price_change > -0.5:
        return 65

    if obv_change > 2:
        return 35

    return 0


# ============================================================
# YENI SINYAL 3: DIRENCTE SATIS EMILIMI
# ============================================================

def resistance_absorption(highs, lows, closes, volumes):
    if len(closes) < 35:
        return 0

    resistance = max(highs[-25:-3])

    if resistance <= 0:
        return 0

    recent_volume = statistics.mean(volumes[-3:])
    normal_volume = statistics.mean(volumes[-25:-3])

    if normal_volume <= 0:
        return 0

    volume_ratio = recent_volume / normal_volume

    near_resistance = (
        closes[-1] >= resistance * 0.985
        or highs[-1] >= resistance * 0.995
    )

    price_stable = closes[-1] >= closes[-4] * 0.995

    candle_position = (
        (closes[-1] - lows[-1]) /
        max(highs[-1] - lows[-1], 1e-12)
    )

    if volume_ratio >= 2 and near_resistance and price_stable:
        return 100

    if volume_ratio >= 1.5 and near_resistance and candle_position > 0.5:
        return 75

    if volume_ratio >= 1.3 and near_resistance:
        return 45

    return 0


# ============================================================
# YENI SINYAL 4: YUKSELEN DIPLER
# ============================================================

def higher_lows(lows):
    if len(lows) < 35:
        return 0

    swing_lows = []

    for i in range(len(lows) - 20, len(lows) - 2):
        if (
            lows[i] < lows[i - 1]
            and lows[i] <= lows[i + 1]
        ):
            swing_lows.append(lows[i])

    if len(swing_lows) < 2:
        return 0

    last = swing_lows[-3:]

    if len(last) >= 3:
        if last[-1] > last[-2] > last[-3]:
            return 100

    if last[-1] > last[-2]:
        return 65

    return 0


# ============================================================
# 50 TEKNIK KONTROL - HER ZAMAN DILIMINDE 10 KONTROL
# ============================================================

def technical_analysis(candles):

    opens = [float(x[1]) for x in candles]
    highs = [float(x[2]) for x in candles]
    lows = [float(x[3]) for x in candles]
    closes = [float(x[4]) for x in candles]
    volumes = [float(x[5]) for x in candles]
    buy_volumes = [float(x[9]) for x in candles]

    if len(closes) < 60:
        return None

    price = closes[-1]

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    ema50 = ema(closes, 50)

    rsi_value = rsi(closes)

    macd_fast = ema(closes, 12)
    macd_slow = ema(closes, 26)
    macd_value = macd_fast - macd_slow

    previous_macd = (
        ema(closes[:-1], 12) -
        ema(closes[:-1], 26)
    )

    bb = bollinger(closes)

    obv = obv_values(closes, volumes)

    volume_average = statistics.mean(volumes[-21:-1])
    volume_ratio = (
        volumes[-1] / volume_average
        if volume_average else 0
    )

    roc = safe_pct(price, closes[-11])

    atr_value = atr(highs, lows, closes)
    atr_percent = atr_value / price * 100 if price else 0

    stoch = stochastic_rsi(closes)

    # Futures mum verisindeki taker buy volume kullanilir.
    # Bu deger gercek emir defteri CVD'si degildir.
    buy_ratio = (
        sum(buy_volumes[-10:]) /
        max(sum(volumes[-10:]), 1e-12)
    )

    obv_positive = obv[-1] > obv[-10]

    checks = [
        price > ema21,                     # 1 EMA trend
        ema9 > ema21,                      # 2 EMA kesisimi
        ema21 > ema50,                     # 3 Uzun trend
        rsi_value > 50 and rsi_value < 75, # 4 RSI
        macd_value > 0,                    # 5 MACD
        macd_value > previous_macd,        # 6 MACD ivme
        bb is not None and price > bb[1],  # 7 Bollinger
        stoch > 0.5,                       # 8 Stoch RSI
        roc > 0,                            # 9 Momentum
        obv_positive and volume_ratio > 1, # 10 OBV/hacim
    ]

    short_checks = [
        price < ema21,
        ema9 < ema21,
        ema21 < ema50,
        rsi_value < 50 and rsi_value > 25,
        macd_value < 0,
        macd_value < previous_macd,
        bb is not None and price < bb[1],
        stoch < 0.5,
        roc < 0,
        not obv_positive and volume_ratio > 1,
    ]

    long_score = sum(checks) * 10
    short_score = sum(short_checks) * 10

    # Yeni erken akumulasyon faktorlari
    squeeze = bollinger_squeeze(closes)
    divergence = obv_divergence(lows, closes, volumes)
    absorption = resistance_absorption(
        highs, lows, closes, volumes
    )
    rising_lows = higher_lows(lows)

    early_score = (
        squeeze * 0.20 +
        divergence * 0.25 +
        absorption * 0.25 +
        rising_lows * 0.30
    )

    return {
        "price": price,
        "long": long_score,
        "short": short_score,
        "rsi": round(rsi_value, 1),
        "volume_ratio": round(volume_ratio, 2),
        "atr_percent": round(atr_percent, 2),
        "buy_ratio": round(buy_ratio * 100, 1),
        "squeeze": squeeze,
        "divergence": divergence,
        "absorption": absorption,
        "higher_lows": rising_lows,
        "early": round(early_score),
    }


# ============================================================
# FUTURES EK VERILERI
# ============================================================

def get_futures_extra(symbol):
    result = {
        "funding": None,
        "oi": None,
        "oi_change": None,
    }

    try:
        data = api_get(
            "/fapi/v1/premiumIndex",
            {"symbol": symbol}
        )
        result["funding"] = float(data["lastFundingRate"]) * 100
    except Exception:
        pass

    try:
        data = api_get(
            "/fapi/v1/openInterest",
            {"symbol": symbol}
        )
        result["oi"] = float(data["openInterest"])
    except Exception:
        pass

    try:
        data = api_get(
            "/futures/data/openInterestHist",
            {
                "symbol": symbol,
                "period": "15m",
                "limit": 2
            }
        )

        if len(data) >= 2:
            old = float(data[0]["sumOpenInterest"])
            new = float(data[-1]["sumOpenInterest"])

            if old:
                result["oi_change"] = round(
                    safe_pct(new, old), 2
                )
    except Exception:
        pass

    return result


# ============================================================
# SEMBOL LISTESI - HACME GORE ILK 100
# ============================================================

def get_top_symbols():

    exchange = api_get("/fapi/v1/exchangeInfo")

    valid_symbols = {
        item["symbol"]
        for item in exchange["symbols"]
        if (
            item["quoteAsset"] == "USDT"
            and item["status"] == "TRADING"
            and item["contractType"] == "PERPETUAL"
        )
    }

    tickers = api_get("/fapi/v1/ticker/24hr")

    ranked = []

    for ticker in tickers:
        symbol = ticker.get("symbol")

        if symbol not in valid_symbols:
            continue

        try:
            volume = float(ticker["quoteVolume"])
            price = float(ticker["lastPrice"])

            if volume > 0 and price > 0:
                ranked.append((symbol, volume))
        except (ValueError, TypeError, KeyError):
            continue

    ranked.sort(key=lambda x: x[1], reverse=True)

    return [x[0] for x in ranked[:TOP_N]]


# ============================================================
# TEK COIN TARAMASI
# ============================================================

def scan_symbol(symbol):

    data = {
        "symbol": symbol,
        "timeframes": {},
        "error": None,
    }

    try:
        for tf in TIMEFRAMES:

            candles = api_get(
                "/fapi/v1/klines",
                {
                    "symbol": symbol,
                    "interval": tf,
                    "limit": KLINE_LIMIT,
                }
            )

            analysis = technical_analysis(candles)

            if analysis is None:
                raise Exception(f"{tf} mum verisi yetersiz")

            data["timeframes"][tf] = analysis

        data["extra"] = get_futures_extra(symbol)

        # Ana LONG/SHORT puani
        data["long_score"] = round(sum(
            data["timeframes"][tf]["long"] *
            TF_WEIGHTS[tf] / 100
            for tf in TIMEFRAMES
        ))

        data["short_score"] = round(sum(
            data["timeframes"][tf]["short"] *
            TF_WEIGHTS[tf] / 100
            for tf in TIMEFRAMES
        ))

        # Erken akumulasyon: 5m, 15m ve 1h
        early_tfs = ["5m", "15m", "1h"]

        data["early_score"] = round(sum(
            data["timeframes"][tf]["early"]
            for tf in early_tfs
        ) / len(early_tfs))

        # Yükselen dipler 3 zaman diliminde kontrol edilir
        data["rising_count"] = sum(
            data["timeframes"][tf]["higher_lows"] >= 65
            for tf in early_tfs
        )

        data["signal"] = (
            "GUCLU LONG HAZIRLIK"
            if data["long_score"] >= 75
            and data["early_score"] >= 65
            else "LONG ADAYI"
            if data["long_score"] >= 65
            else "GUCLU SHORT"
            if data["short_score"] >= 75
            else "IZLEME"
        )

        return data

    except Exception as exc:
        data["error"] = str(exc)
        return data


# ============================================================
# RAPORLAMA
# ============================================================

def signal_text(item):

    if item.get("error"):
        return None

    t = item["timeframes"]
    extra = item.get("extra", {})

    lines = [
        f"#{item['symbol']}",
        f"Sinyal: {item['signal']}",
        f"LONG: {item['long_score']}/100",
        f"SHORT: {item['short_score']}/100",
        f"ERKEN AKUMULASYON: {item['early_score']}/100",
        "",
        "YENI SINYAL FAKTORLERI:",
    ]

    for tf in ["5m", "15m", "1h"]:
        x = t[tf]

        lines.append(
            f"{tf}: BB {x['squeeze']} | "
            f"OBV {x['divergence']} | "
            f"Emilim {x['absorption']} | "
            f"Yukselen dip {x['higher_lows']}"
        )

    lines.extend([
        "",
        f"Yukselen dip: {item['rising_count']}/3 zaman dilimi",
        f"RSI (1h): {t['1h']['rsi']}",
        f"Hacim carpani (1h): {t['1h']['volume_ratio']}",
        f"Taker alis orani (1h): {t['1h']['buy_ratio']}%",
        f"Funding: {extra.get('funding')}",
        f"OI: {extra.get('oi')}",
        f"OI degisimi 15m: {extra.get('oi_change')}%",
        "--------------------------",
    ])

    return "\n".join(lines)


def send_telegram(message):

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Telegram ayarlari bulunamadi. Rapor ekrana yazdirildi.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    try:
        response = requests.post(
            url,
            data={
                "chat_id": chat_id,
                "text": message,
                "disable_web_page_preview": True,
            },
            timeout=20,
        )
        response.raise_for_status()

    except Exception as exc:
        print("Telegram gonderim hatasi:", exc)


# ============================================================
# ANA PROGRAM
# ============================================================

def main():

    start = datetime.now(timezone.utc)

    print("=" * 55)
    print("KRIPTO FUTURES RADAR 50 - V", VERSION)
    print("Tarama:", start.isoformat())
    print("=" * 55)

    try:
        symbols = get_top_symbols()
    except Exception as exc:
        print("SEMBOL LISTESI ALINAMADI:", exc)
        return

    print("Taranacak parite:", len(symbols))

    results = []
    errors = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {
            executor.submit(scan_symbol, symbol): symbol
            for symbol in symbols
        }

        completed = 0

        for future in as_completed(futures):
            completed += 1

            try:
                result = future.result()

                if result["error"]:
                    errors += 1
                    print(
                        f"[{completed}/{len(symbols)}] "
                        f"{result['symbol']} HATA: {result['error']}"
                    )
                else:
                    results.append(result)
                    print(
                        f"[{completed}/{len(symbols)}] "
                        f"{result['symbol']} "
                        f"LONG:{result['long_score']} "
                        f"ERKEN:{result['early_score']}"
                    )

            except Exception as exc:
                errors += 1
                print("Tarama hatasi:", exc)

    results.sort(
        key=lambda x: (
            x["early_score"],
            x["long_score"]
        ),
        reverse=True
    )

    valid_count = len(results)

    if not results:
        print("Gecerli tarama sonucu bulunamadi.")
        return

    # Ilk 15 sonucu rapora al
    report_items = results[:15]

    report = [
        "FUTURES RADAR 50",
        f"Tarih: {datetime.now(timezone.utc).isoformat()}",
        f"Taranan: {len(symbols)}",
        f"Basarili: {valid_count}",
        f"Hata: {errors}",
        "",
        "EN GUCLU ADAYLAR",
        "==========================",
    ]

    for item in report_items:
        text_block = signal_text(item)

        if text_block:
            report.append(text_block)

    message = "\n".join(report)

    print("\n" + message)

    # Telegram mesaj limiti icin parcalama
    chunks = []
    current = ""

    for line in message.splitlines(keepends=True):
        if len(current) + len(line) > 3500:
            chunks.append(current)
            current = ""
        current += line

    if current:
        chunks.append(current)

    for chunk in chunks:
        send_telegram(chunk)

    print("\nTarama tamamlandi.")
    print("Basarili:", valid_count)
    print("Hatali:", errors)
    print("Sure:", datetime.now(timezone.utc) - start)


if __name__ == "__main__":
    main()

