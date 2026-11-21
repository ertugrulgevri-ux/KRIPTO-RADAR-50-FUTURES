
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
    "User-Agent": "FUTURE-RADAR-50/3.0"
})


def get_binance_data(endpoint, attempts=4):
    """Binance Futures API verisini kontrollu alir."""

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
            return response.json()

        except (requests.RequestException, ValueError) as error:
            if attempt == attempts - 1:
                raise RuntimeError(
                    f"Binance verisi alinamadi: {error}"
                )

            time.sleep(2 * (attempt + 1))

    raise RuntimeError("Binance baglantisi basarisiz.")


def safe_float(value):
    try:
        number = float(value)

        if not os.path.exists(OUTPUT_FILE):
            pass

        if number != number or abs(number) == float("inf"):
            return 0.0

        return number

    except (TypeError, ValueError):
        return 0.0


def classify_asset(item):
    """Binance underlyingType ve subtype bilgilerini kullanir."""

    underlying_type = str(
        item.get("underlyingType", "")
    ).upper()

    subtypes = item.get("underlyingSubType", [])

    if not isinstance(subtypes, list):
        subtypes = []

    subtype_text = " ".join(
        str(value).upper() for value in subtypes
    )

    if any(x in subtype_text for x in (
        "STOCK", "EQUITY", "SHARES"
    )):
        return "STOCK"

    if any(x in subtype_text for x in (
        "METAL", "ENERGY", "COMMODITY"
    )):
        return "COMMODITY"

    if "FOREX" in subtype_text or "CURRENCY" in subtype_text:
        return "FOREX"

    if "BOND" in subtype_text:
        return "BOND"

    if underlying_type == "COIN":
        return "CRYPTO"

    if underlying_type == "INDEX":
        return "INDEX"

    return "UNCLASSIFIED"


def get_filter_value(item, filter_type, key):
    for entry in item.get("filters", []):
        if entry.get("filterType") == filter_type:
            return safe_float(entry.get(key))

    return 0.0


def main():

    print("=" * 60)
    print("          FUTURE RADAR 50")
    print("     BINANCE USD-M FUTURES TARAMASI")
    print("=" * 60)

    # 1. Sozlesme bilgileri
    print("\n[1/5] Binance sozlesmeleri aliniyor...")

    exchange_info = get_binance_data(
        "/fapi/v1/exchangeInfo"
    )

    instruments = exchange_info.get("symbols", [])

    if not isinstance(instruments, list) or not instruments:
        raise RuntimeError("Sozlesme verisi bulunamadi.")

    print("Toplam sozlesme:", len(instruments))

    # 2. 24 saatlik piyasa verileri
    print("\n[2/5] Piyasa verileri aliniyor...")

    tickers = get_binance_data(
        "/fapi/v1/ticker/24hr"
    )

    if not isinstance(tickers, list):
        raise RuntimeError("Ticker verisi formati hatali.")

    ticker_map = {
        item["symbol"]: item
        for item in tickers
        if isinstance(item, dict) and item.get("symbol")
    }

    print("Ticker sayisi:", len(ticker_map))

    # 3. Uygun sozlesmeleri filtrele
    print("\n[3/5] Aktif USDT perpetual sozlesmeler seciliyor...")

    assets = []
    excluded = 0
    missing_ticker = 0

    for item in instruments:

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
        open_price = safe_float(ticker.get("openPrice"))
        base_volume = safe_float(ticker.get("volume"))
        quote_volume = safe_float(ticker.get("quoteVolume"))

        if price <= 0 or quote_volume <= 0:
            excluded += 1
            continue

        change = safe_float(
            ticker.get("priceChangePercent")
        )

        assets.append({
            "symbol": symbol,
            "baseAsset": item.get("baseAsset"),
            "quoteAsset": item.get("quoteAsset"),
            "settleAsset": item.get("marginAsset"),
            "contractType": item.get("contractType"),
            "exchange": "BINANCE",
            "underlyingType": item.get("underlyingType", ""),
            "underlyingSubType": item.get(
                "underlyingSubType", []
            ),
            "category": classify_asset(item),
            "price": price,
            "open24h": open_price,
            "change24hPercent": change,
            "high24h": safe_float(ticker.get("highPrice")),
            "low24h": safe_float(ticker.get("lowPrice")),
            "baseVolume24h": base_volume,
            "quoteVolume24h": quote_volume,
            "tickSize": get_filter_value(
                item, "PRICE_FILTER", "tickSize"
            ),
            "lotSize": get_filter_value(
                item, "LOT_SIZE", "stepSize"
            ),
            "minSize": get_filter_value(
                item, "LOT_SIZE", "minQty"
            )
        })

    if not assets:
        raise RuntimeError(
            "Uygun USDT perpetual sozlesme bulunamadi."
        )

    # 4. Hacme gore siralama
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
        if x["category"] not in (
            "CRYPTO", "UNCLASSIFIED"
        )
    ]

    unclassified = [
        x for x in assets
        if x["category"] == "UNCLASSIFIED"
    ]

    # 5. Sonuc dosyasi
    print("\n[5/5] Sonuc dosyasi hazirlaniyor...")

    output = {
        "project": "FUTURE RADAR 50",
        "market": "BINANCE USD-M USDT PERPETUAL",
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
            "total_exchange_symbols": len(instruments),
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
    print("\n" + "=" * 60)
    print("BINANCE TARAMASI TAMAMLANDI")
    print("=" * 60)

    print("Uygun toplam varlik:", len(assets))
    print("Kripto:", len(crypto))
    print("Kripto disi:", len(non_crypto))
    print("Siniflandirilamayan:", len(unclassified))
    print("Eksik ticker:", missing_ticker)

    print("\nEN YUKSEK HACIMLI ILK 20")

    for index, asset in enumerate(assets[:20], 1):
        print(
            f"{index:02d}. {asset['symbol']:<20}"
            f" {asset['category']:<14}"
            f" Hacim: {asset['quoteVolume24h']:,.0f} USDT"
        )

    print("\nDosya olusturuldu:", OUTPUT_FILE)
    print("FUTURE RADAR 50 piyasa kesfi tamamlandi.")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\nRADAR HATASI:", str(error))
        raise

    finally:
        SESSION.close()
