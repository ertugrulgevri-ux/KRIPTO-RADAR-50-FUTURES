
import json
import time
import os
from datetime import datetime, timezone

import requests

# ==========================================
# FUTURE RADAR 50
# BINANCE USD-M FUTURES MARKET DISCOVERY
# ==========================================

BASE_URL = "https://fapi.binance.com"
OUTPUT_FILE = "futures_universe.json"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "FUTURE-RADAR-50/1.0"
})


def get_binance_data(endpoint, attempts=4):
    """Binance API verisini kontrollü şekilde alır."""

    url = BASE_URL + endpoint

    for attempt in range(attempts):
        try:
            response = SESSION.get(url, timeout=30)

            if response.status_code in (418, 429):
                wait = min(5 * (attempt + 1), 30)
                print("API limit uyarisi. Bekleniyor:", wait)
                time.sleep(wait)
                continue

            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict) and "code" in data:
                raise RuntimeError(
                    "Binance API: " + str(data)
                )

            return data

        except (requests.RequestException, ValueError) as error:
            if attempt == attempts - 1:
                raise RuntimeError(
                    f"API verisi alinamadi: {error}"
                )

            time.sleep(2 * (attempt + 1))

    raise RuntimeError("API baglantisi basarisiz.")


def safe_float(value):
    try:
        number = float(value)
        if number != number or abs(number) == float("inf"):
            return 0.0
        return number
    except (TypeError, ValueError):
        return 0.0


def classify_asset(item):
    """Binance'in sözleşme metadata bilgisine göre sınıflandırır."""

    underlying_type = str(
        item.get("underlyingType", "")
    ).upper()

    subtypes = item.get("underlyingSubType", [])

    if isinstance(subtypes, str):
        subtypes = [subtypes]

    subtype_text = " ".join(
        str(x).upper() for x in subtypes
    )

    crypto_types = {"COIN", "CRYPTO", "CRYPTocurrency".upper()}

    non_crypto_types = {
        "EQUITY", "STOCK", "COMMODITY",
        "INDEX", "ETF", "FOREX"
    }

    if underlying_type in crypto_types:
        return "CRYPTO"

    if underlying_type in non_crypto_types:
        return "NON_CRYPTO"

    crypto_tags = {
        "LAYER-1", "LAYER-2", "DEFI",
        "MEME", "NFT", "GAMING",
        "PAYMENT", "STORAGE"
    }

    if any(tag in subtype_text for tag in crypto_tags):
        return "CRYPTO"

    return "UNCLASSIFIED"


def main():

    print("=" * 55)
    print("       FUTURE RADAR 50")
    print("       BINANCE FUTURES TARAMASI")
    print("=" * 55)

    # 1. Sözleşme bilgileri
    print("\n[1/5] Binance sozlesmeleri aliniyor...")

    exchange = get_binance_data(
        "/fapi/v1/exchangeInfo"
    )

    if not isinstance(exchange.get("symbols"), list):
        raise RuntimeError("ExchangeInfo formati hatali.")

    symbols = exchange["symbols"]

    print("Toplam sozlesme:", len(symbols))

    # 2. 24 saatlik piyasa verileri
    print("\n[2/5] 24 saatlik piyasa verileri aliniyor...")

    tickers = get_binance_data(
        "/fapi/v1/ticker/24hr"
    )

    if not isinstance(tickers, list):
        raise RuntimeError("Ticker verisi liste degil.")

    ticker_map = {
        item["symbol"]: item
        for item in tickers
        if isinstance(item, dict) and item.get("symbol")
    }

    print("Ticker sayisi:", len(ticker_map))

    # 3. Uygun sözleşmeleri filtrele
    print("\n[3/5] Aktif USDT perpetual sozlesmeler seciliyor...")

    assets = []
    excluded = 0
    missing_ticker = 0

    for item in symbols:

        if item.get("status") != "TRADING":
            excluded += 1
            continue

        if item.get("contractType") != "PERPETUAL":
            excluded += 1
            continue

        if item.get("quoteAsset") != "USDT":
            excluded += 1
            continue

        if item.get("marginAsset") != "USDT":
            excluded += 1
            continue

        symbol = item.get("symbol", "")
        ticker = ticker_map.get(symbol)

        if ticker is None:
            missing_ticker += 1
            continue

        price = safe_float(ticker.get("lastPrice"))
        volume = safe_float(ticker.get("quoteVolume"))

        if price <= 0 or volume <= 0:
            excluded += 1
            continue

        assets.append({
            "symbol": symbol,
            "baseAsset": item.get("baseAsset"),
            "quoteAsset": item.get("quoteAsset"),
            "marginAsset": item.get("marginAsset"),
            "contractType": item.get("contractType"),
            "underlyingType": item.get("underlyingType"),
            "underlyingSubType": item.get(
                "underlyingSubType", []
            ),
            "category": classify_asset(item),
            "price": price,
            "change24hPercent": safe_float(
                ticker.get("priceChangePercent")
            ),
            "high24h": safe_float(
                ticker.get("highPrice")
            ),
            "low24h": safe_float(
                ticker.get("lowPrice")
            ),
            "baseVolume24h": safe_float(
                ticker.get("volume")
            ),
            "quoteVolume24h": volume,
            "tradeCount24h": int(
                safe_float(ticker.get("count"))
            )
        })

    if not assets:
        raise RuntimeError(
            "Uygun USDT perpetual sozlesme bulunamadi."
        )

    # 4. Hacme göre sıralama
    print("\n[4/5] Hacim siralamasi yapiliyor...")

    assets.sort(
        key=lambda x: x["quoteVolume24h"],
        reverse=True
    )

    crypto = [
        x for x in assets
        if x["category"] == "CRYPTO"
    ]

    non_crypto = [
        x for x in assets
        if x["category"] == "NON_CRYPTO"
    ]

    unclassified = [
        x for x in assets
        if x["category"] == "UNCLASSIFIED"
    ]

    # 5. Sonuç dosyası
    print("\n[5/5] Sonuc dosyasi hazirlaniyor...")

    output = {
        "project": "FUTURE RADAR 50",
        "market": "BINANCE USD-M FUTURES",
        "scan_time": datetime.now(
            timezone.utc
        ).isoformat(),
        "filters": {
            "status": "TRADING",
            "contractType": "PERPETUAL",
            "quoteAsset": "USDT",
            "marginAsset": "USDT"
        },
        "summary": {
            "total_exchange_symbols": len(symbols),
            "total_tickers": len(ticker_map),
            "eligible_assets": len(assets),
            "crypto_count": len(crypto),
            "non_crypto_count": len(non_crypto),
            "unclassified_count": len(unclassified),
            "excluded_count": excluded,
            "missing_ticker_count": missing_ticker
        },
        "top50_by_volume": assets[:50],
        "assets": assets
    }

    temp_file = OUTPUT_FILE + ".tmp"

    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        )

    os.replace(temp_file, OUTPUT_FILE)

    # Rapor
    print("\n" + "=" * 55)
    print("TARAMA BASARILI")
    print("=" * 55)

    print("Uygun toplam varlik:", len(assets))
    print("Kripto:", len(crypto))
    print("Kripto disi:", len(non_crypto))
    print("Siniflandirilmamis:", len(unclassified))
    print("Eksik ticker:", missing_ticker)

    print("\nEN YUKSEK HACIMLI ILK 20")

    for index, asset in enumerate(assets[:20], 1):
        print(
            f"{index:02d}. {asset['symbol']:<16}"
            f" {asset['category']:<14}"
            f" Hacim: {asset['quoteVolume24h']:,.0f} USDT"
        )

    print("\nDosya olusturuldu:", OUTPUT_FILE)
    print("FUTURE RADAR 50 Piyasa kesfi tamamlandi.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nRADAR HATASI:", str(error))
        raise
    finally:
        SESSION.close()
