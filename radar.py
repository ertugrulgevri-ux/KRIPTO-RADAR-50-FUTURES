
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
    url = BASE_URL + endpoint

    for attempt in range(attempts):
        try:
            response = SESSION.get(
                url,
                params=params,
                timeout=25
            )

            if response.status_code in (418, 429):
                time.sleep(2 ** attempt)
                continue

            response.raise_for_status()
            result = response.json()

            if result.get("code") != "0":
                raise RuntimeError(
                    f"OKX API hatasi: {result.get('msg')}"
                )

            return result.get("data", [])

        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError("OKX verisi alinamadi.")


def safe_float(value, default=0.0):
    try:
        number = float(value)
        if number != number or abs(number) == float("inf"):
            return default
        return number
    except (TypeError, ValueError):
        return default


def classify_asset(inst_category):
    mapping = {
        "1": "CRYPTO",
        "3": "STOCK",
        "4": "COMMODITY",
        "5": "FOREX",
        "6": "BOND"
    }

    return mapping.get(str(inst_category), "UNCLASSIFIED")


def get_filter_value(filters, filter_type, field):
    for item in filters:
        if item.get("filterType") == filter_type:
            return item.get(field, "")
    return ""


def main():
    print("=" * 56)
    print("          FUTURE RADAR 50")
    print("       OKX USDT FUTURES TARAMASI")
    print("=" * 56)

    print("\n[1/5] OKX sozlesmeleri aliniyor...")

    instruments = get_okx_data(
        "/api/v5/public/instruments",
        {"instType": "SWAP"}
    )

    print(f"Bulunan SWAP sozlesmesi: {len(instruments)}")

    print("\n[2/5] 24 saatlik piyasa verileri aliniyor...")

    tickers = get_okx_data(
        "/api/v5/market/tickers",
        {"instType": "SWAP"}
    )

    ticker_map = {
        item.get("instId"): item
        for item in tickers
    }

    print(f"Bulunan ticker: {len(tickers)}")

    print("\n[3/5] USDT perpetual sozlesmeler filtreleniyor...")

    assets = []
    excluded = 0

    for instrument in instruments:
        inst_id = instrument.get("instId", "")

        if instrument.get("state") != "live":
            excluded += 1
            continue

        if instrument.get("instType") != "SWAP":
            excluded += 1
            continue

        if instrument.get("settleCcy") != "USDT":
            excluded += 1
            continue

        if instrument.get("ctType") != "linear":
            excluded += 1
            continue

        if instrument.get("expTime", ""):
            excluded += 1
            continue

        ticker = ticker_map.get(inst_id)

        if not ticker:
            excluded += 1
            continue

        price = safe_float(ticker.get("last"))
        volume = safe_float(ticker.get("volCcy24h"))

        quote_volume = volume * price

        if price <= 0 or quote_volume <= 0:
            excluded += 1
            continue

        category = classify_asset(
            instrument.get("instCategory", "")
        )

        assets.append({
            "symbol": inst_id,
            "instId": inst_id,
            "baseAsset": instrument.get("baseCcy", ""),
            "quoteAsset": instrument.get("quoteCcy", ""),
            "settleAsset": instrument.get("settleCcy", ""),
            "exchange": "OKX",
            "category": category,
            "price": price,
            "open24h": safe_float(ticker.get("open24h")),
            "change24hPercent": safe_float(
                ticker.get("change24h")
            ),
            "high24h": safe_float(ticker.get("high24h")),
            "low24h": safe_float(ticker.get("low24h")),
            "baseVolume24h": safe_float(ticker.get("vol24h")),
            "quoteVolume24h": quote_volume,
            "tickSize": get_filter_value(
                instrument.get("filters", []),
                "PRICE_FILTER",
                "tickSize"
            ),
            "lotSize": instrument.get("lotSz", ""),
            "minSize": instrument.get("minSz", "")
        })

    print(f"Uygun sozlesme: {len(assets)}")
    print(f"Filtrelenen: {excluded}")

    print("\n[4/5] Varliklar siniflandiriliyor...")

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
        if x["category"] in (
            "STOCK", "COMMODITY", "FOREX", "BOND"
        )
    ]

    unclassified = [
        x for x in assets
        if x["category"] == "UNCLASSIFIED"
    ]

    print(f"CRYPTO: {len(crypto)}")
    print(f"NON-CRYPTO: {len(non_crypto)}")
    print(f"UNCLASSIFIED: {len(unclassified)}")

    print("\n[5/5] Sonuc dosyasi olusturuluyor...")

    output = {
        "exchange": "OKX",
        "market": "USDT_LINEAR_PERPETUAL",
        "scan_time": datetime.now(
            timezone.utc
        ).isoformat(),
        "summary": {
            "total_assets": len(assets),
            "crypto": len(crypto),
            "non_crypto": len(non_crypto),
            "unclassified": len(unclassified)
        },
        "filters": {
            "instType": "SWAP",
            "settleCcy": "USDT",
            "ctType": "linear",
            "state": "live",
            "expTime": "empty"
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
            indent=2
        )

    os.replace(temp_file, OUTPUT_FILE)

    print("\n" + "=" * 56)
    print("TARAMA TAMAMLANDI")
    print("=" * 56)
    print(f"Toplam uygun sozlesme: {len(assets)}")
    print(f"Kripto: {len(crypto)}")
    print(f"Kripto disi: {len(non_crypto)}")
    print(f"Siniflandirilmamis: {len(unclassified)}")
    print(f"Dosya: {OUTPUT_FILE}")

    print("\nEN YUKSEK HACIMLI 10 SOZLESME:")

    for index, asset in enumerate(assets[:10], 1):
        print(
            f"{index}. {asset['instId']} | "
            f"{asset['category']} | "
            f"{asset['quoteVolume24h']:,.0f} USDT"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nRADAR HATASI: {error}")
        raise
