
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone

BASE_URL = "https://fapi.binance.com"

TOP_N = 100
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]

def get_json(path, params=None):
    response = requests.get(BASE_URL + path, params=params, timeout=20)
    response.raise_for_status()
    return response.json()

def get_top_symbols():
    exchange_info = get_json("/fapi/v1/exchangeInfo")
    tickers = get_json("/fapi/v1/ticker/24hr")

    valid = {
        item["symbol"]
        for item in exchange_info["symbols"]
        if item["status"] == "TRADING"
        and item["contractType"] == "PERPETUAL"
        and item["quoteAsset"] == "USDT"
    }

    rows = [
        item for item in tickers
        if item["symbol"] in valid
    ]

    rows.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    return rows[:TOP_N]

def get_klines(symbol, interval, limit=250):
    data = get_json(
        "/fapi/v1/klines",
        {"symbol": symbol, "interval": interval, "limit": limit}
    )

    columns = [
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]

    df = pd.DataFrame(data, columns=columns)

    for col in ["open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def get_futures_data(symbol):
    oi = get_json("/fapi/v1/openInterest", {"symbol": symbol})
    funding = get_json("/fapi/v1/premiumIndex", {"symbol": symbol})

    return {
        "open_interest": float(oi["openInterest"]),
        "funding_rate": float(funding["lastFundingRate"]),
        "mark_price": float(funding["markPrice"]),
        "index_price": float(funding["indexPrice"])
    }

def basic_indicators(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    obv = (np.sign(delta).fillna(0) * volume).cumsum()
    volume_avg = volume.rolling(20).mean()

    atr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1).rolling(14).mean()

    return {
        "rsi": float(rsi.iloc[-1]),
        "ema20": float(ema20.iloc[-1]),
        "ema50": float(ema50.iloc[-1]),
        "obv": float(obv.iloc[-1]),
        "volume_ratio": float(volume.iloc[-1] / volume_avg.iloc[-1]),
        "atr_percent": float(atr.iloc[-1] / close.iloc[-1] * 100),
        "price": float(close.iloc[-1])
    }

def scan_symbol(symbol):
    result = {
        "symbol": symbol,
        "timeframes": {},
        "futures": {},
        "status": "OK"
    }

    try:
        for timeframe in TIMEFRAMES:
            df = get_klines(symbol, timeframe)
            result["timeframes"][timeframe] = basic_indicators(df)
            time.sleep(0.05)

        result["futures"] = get_futures_data(symbol)

    except Exception as error:
        result["status"] = f"ERROR: {error}"

    return result

def main():
    print("KRIPTO RADAR 50 FUTURES")
    print("Tarama baslangici:", datetime.now(timezone.utc).isoformat())

    symbols = get_top_symbols()
    print("Taranacak parite:", len(symbols))

    results = []

    for index, item in enumerate(symbols, start=1):
        symbol = item["symbol"]
        print(f"{index}/{len(symbols)} - {symbol}")

        result = scan_symbol(symbol)
        result["quote_volume_24h"] = float(item["quoteVolume"])
        results.append(result)

    successful = [r for r in results if r["status"] == "OK"]

    print("\nTarama tamamlandi.")
    print("Basarili:", len(successful))
    print("Hatali:", len(results) - len(successful))

    for result in successful[:10]:
        print(
            result["symbol"],
            "RSI(15m):",
            round(result["timeframes"]["15m"]["rsi"], 2),
            "Funding:",
            result["futures"]["funding_rate"]
        )

if __name__ == "__main__":
    main()

