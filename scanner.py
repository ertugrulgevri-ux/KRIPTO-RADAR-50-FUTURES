
import json
import time
from pathlib import Path

from market_data import fetch_all_timeframes
from indicator_engine import analyze_timeframes


UNIVERSE_FILE = "futures_universe.json"
OUTPUT_FILE = "futures_scan_results.json"

# İlk denemede yalnızca 5 sözleşme taranacak.
SCAN_LIMIT = 5


def find_asset_list(data):
    """JSON içindeki en uzun sözleşme listesini bulur."""
    found = []

    def search(item):
        if isinstance(item, list):
            if item and isinstance(item[0], dict):
                if any(
                    "symbol" in row or "instId" in row
                    for row in item
                ):
                    found.append(item)

            for child in item:
                search(child)

        elif isinstance(item, dict):
            for child in item.values():
                search(child)

    search(data)

    if not found:
        raise RuntimeError(
            "futures_universe.json içinde sözleşme listesi bulunamadı."
        )

    return max(found, key=len)


def get_volume(asset):
    for key in (
        "quoteVolume24h",
        "quote_volume24h",
        "quoteVolume",
        "volume24h",
    ):
        if key in asset:
            try:
                return float(asset[key])
            except (TypeError, ValueError):
                pass
    return 0.0


def load_symbols():
    path = Path(UNIVERSE_FILE)

    if not path.exists():
        raise FileNotFoundError(
            f"{UNIVERSE_FILE} bulunamadı. Önce radar.py çalıştırılmalı."
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    assets = find_asset_list(data)

    # Hacmi yüksek olan sözleşmeler önce taranır.
    assets.sort(key=get_volume, reverse=True)

    selected = []

    for asset in assets:
        symbol = asset.get("symbol") or asset.get("instId")

        if symbol and symbol not in selected:
            selected.append(symbol)

        if len(selected) >= SCAN_LIMIT:
            break

    if not selected:
        raise RuntimeError("Taranacak sözleşme bulunamadı.")

    return selected, assets


def run_scan():
    print("=" * 60)
    print("FUTURE RADAR 50 - COKLU SOZLESME TARAMASI")
    print("=" * 60)

    symbols, assets = load_symbols()

    asset_info = {
        (item.get("symbol") or item.get("instId")): item
        for item in assets
    }

    print(f"\nPiyasada bulunan sözleşme sayısı: {len(assets)}")
    print(f"Bu denemede taranacak sözleşme: {len(symbols)}")

    results = []
    errors = []

    for index, symbol in enumerate(symbols, start=1):
        print("\n" + "-" * 60)
        print(f"[{index}/{len(symbols)}] {symbol} analiz ediliyor...")

        try:
            timeframe_data = fetch_all_timeframes(symbol)

            result = analyze_timeframes(timeframe_data)

            info = asset_info.get(symbol, {})

            record = {
                "symbol": symbol,
                "category": info.get("category", "UNCLASSIFIED"),
                "price": info.get("price"),
                "long_score": round(float(result["long_score"]), 2),
                "short_score": round(float(result["short_score"]), 2),
                "long_label": result["long_label"],
                "short_label": result["short_label"],
                "complete_timeframes": result["complete_timeframes"],
                "available_timeframe_weight": result[
                    "available_timeframe_weight"
                ],
            }

            results.append(record)

            print(
                f"LONG: {record['long_score']} "
                f"({record['long_label']})"
            )
            print(
                f"SHORT: {record['short_score']} "
                f"({record['short_label']})"
            )

        except Exception as error:
            errors.append({
                "symbol": symbol,
                "error": str(error),
            })
            print(f"HATA: {error}")

        time.sleep(1)

    long_ranking = sorted(
        results,
        key=lambda item: item["long_score"],
        reverse=True,
    )

    short_ranking = sorted(
        results,
        key=lambda item: item["short_score"],
        reverse=True,
    )

    output = {
        "project": "FUTURE RADAR 50",
        "scanned_count": len(results),
        "failed_count": len(errors),
        "requested_count": len(symbols),
        "long_ranking": long_ranking,
        "short_ranking": short_ranking,
        "errors": errors,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("TARAMA TAMAMLANDI")
    print("=" * 60)
    print(f"Başarılı analiz: {len(results)}")
    print(f"Başarısız analiz: {len(errors)}")
    print(f"Sonuç dosyası: {OUTPUT_FILE}")

    print("\nEN YÜKSEK LONG PUANLARI")
    for item in long_ranking:
        print(item["symbol"], item["long_score"])

    print("\nEN YÜKSEK SHORT PUANLARI")
    for item in short_ranking:
        print(item["symbol"], item["short_score"])


if __name__ == "__main__":
    run_scan()
