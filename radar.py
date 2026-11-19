
import json
import time
import os
from datetime import datetime, timezone

import requests

# ==========================================
# FUTURE RADAR 50
# OKX USDT FUTURES MARKET DISCOVERY
# ==========================================

BASE_URL = "https://www.okx.com"
OUTPUT_FILE = "futures_universe.json"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "FUTURE-RADAR-50/2.0"
})


def get_okx_data(endpoint, params=None, attempts=4):
    """OKX API verisini kontrollü şekilde alır."""

    url = BASE_URL + endpoint

    for attempt in range(attempts):
        try:
            response = SESSION.get(
                url,
                params=params,
                timeout=30
            )

            if response.status_code in (418, 429):
                wait = min(5 * (attempt + 1), 30)
                print("API limit uyarisi. Bekleniyor:", wait)
                time.sleep(wait)
                continue

            response.raise_for_status()
            result = response.json()

            if result.get("code") != "0":
                raise RuntimeError(
                    "OKX API hatasi: " + str(result)
                )

            return result.get("data", [])

        except (requests.RequestException, ValueError) as error:
            if attempt == attempts - 1:
                raise RuntimeError(
                    f"OKX verisi alinamadi: {error}"
                )

            time.sleep(2 * (attempt + 1))

    raise RuntimeError("OKX baglantisi basarisiz.")


def safe_float(value):
    try:
        number = float(value)

        if number != number or abs(number) == float("inf"):
            return 0.0

        return number

    except (TypeError, ValueError):
        return 0.0


def classify_asset(item):
    """OKX instCategory alanina gore varlik siniflandirir."""

    category = str(
        item.get("instCategory", "")
    )

    categories = {
        "1": "CRYPTO",
        "3": "STOCK",
        "4": "COMMODITY",
        "5": "FOREX",
        "6": "BOND"
    }

    return categories.get(category, "UNCLASSIFIED")


def main():

    print("=" * 60)
    print("          FUTURE RADAR 50")
    print("          OKX FUTURES TARAMASI")
    print("=" * 60)

    # 1. Sozlesme bilgileri
    print("\n[1/5] OKX sozlesmeleri aliniyor...")

    instruments = get_okx_data(
        "/api/v5/public/instruments",
        {"instType": "SWAP"}
    )

    if not isinstance(instruments, list):
        raise RuntimeError("Sozlesme verisi formati hatali.")

    print("Toplam SWAP sozlesmesi:", len(instruments))

    # 2. Piyasa verileri
    print("\n[2/5] 24 saatlik piyasa verileri aliniyor...")

    tickers = get_okx_data(
        "/api/v5/market/tickers",
        {"instType": "SWAP"}
    )

    if not isinstance(tickers, list):
        raise RuntimeError("Ticker verisi formati hatali.")

    ticker_map = {
        item["instId"]: item
        for item in tickers
        if isinstance(item, dict) and item.get("instId")
    }

    print("Ticker sayisi:", len(ticker_map))

    # 3. Uygun sozlesmeleri filtrele
    print("\n[3/5] Aktif USDT perpetual sozlesmeler seciliyor...")

    assets = []
    excluded = 0
    missing_ticker = 0

    for item in instruments:

        if item.get("state") != "live":
            excluded += 1
            continue

        if item.get("instType") != "SWAP":
            excluded += 1
            continue

        if item.get("settleCcy") != "USDT":
            excluded += 1
            continue

        if item.get("ctType") != "linear":
            excluded += 1
            continue

        if item.get("expTime", ""):
            excluded += 1
            continue

        symbol = item.get("instId", "")
        ticker = ticker_map.get(symbol)

        if ticker is None:
            missing_ticker += 1
            continue

        price = safe_float(ticker.get("last"))
        open_price = safe_float(ticker.get("open24h"))

        base_volume = safe_float(
            ticker.get("volCcy24h")
        )

        contract_volume = safe_float(
            ticker.get("vol24h")
        )

        quote_volume = base_volume * price

        if price <= 0 or quote_volume <= 0:
            excluded += 1
            continue

        change = 0.0

        if open_price > 0:
            change = (
                (price - open_price) / open_price
            ) * 100

        assets.append({
            "symbol": symbol,
            "baseAsset": item.get("instId", "").split("-")[0],
            "quoteAsset": "USDT",
            "settleAsset": item.get("settleCcy"),
            "contractType": "PERPETUAL",
            "exchange": "OKX",
            "instCategory": item.get("instCategory", ""),
            "category": classify_asset(item),
            "price": price,
            "change24hPercent": change,
            "high24h": safe_float(ticker.get("high24h")),
            "low24h": safe_float(ticker.get("low24h")),
            "baseVolume24h": base_volume,
            "contractVolume24h": contract_volume,
            "quoteVolume24h": quote_volume,
            "tickSize": safe_float(item.get("tickSz")),
            "lotSize": safe_float(item.get("lotSz")),
            "minSize": safe_float(item.get("minSz"))
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
        if x["category"] != "CRYPTO"
        and x["category"] != "UNCLASSIFIED"
    ]

    unclassified = [
        x for x in assets
        if x["category"] == "UNCLASSIFIED"
    ]

    # 5. Sonuc dosyasi
    print("\n[5/5] Sonuc dosyasi hazirlaniyor...")

    output = {
        "project": "FUTURE RADAR 50",
        "market": "OKX USDT LINEAR SWAP",
        "scan_time": datetime.now(
            timezone.utc
        ).isoformat(),
        "filters": {
            "instType": "SWAP",
            "state": "live",
            "settleCcy": "USDT",
            "ctType": "linear",
            "expTime": ""
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
    print("OKX TARAMASI BASARILI")
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
