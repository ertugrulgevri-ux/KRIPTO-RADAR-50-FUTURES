
import json
import os
import time
import urllib.request
from pathlib import Path

from market_data import fetch_all_timeframes
from indicator_engine import analyze_timeframes


UNIVERSE_FILE = "futures_universe.json"
OUTPUT_FILE = "futures_scan_results.json"

SCAN_LIMIT = 531
TOP_COUNT = 10


def find_asset_list(data):
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


def send_telegram_message(message):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Telegram bilgileri bulunamadı. Mesaj gönderilmedi.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    data = {
        "chat_id": chat_id,
        "text": message,
        "disable_web_page_preview": True,
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))

    if not result.get("ok"):
        raise RuntimeError("Telegram mesaj gönderimi başarısız.")

    print("Telegram raporu başarıyla gönderildi.")
    return True


def create_telegram_report(output):
    long_ranking = output["long_ranking"][:TOP_COUNT]
    short_ranking = output["short_ranking"][:TOP_COUNT]

    lines = [
        "FUTURE RADAR 50",
        "OKX FUTURES PIYASA RAPORU",
        "",
        f"Taranan: {output['scanned_count']}",
        f"Basarisiz: {output['failed_count']}",
        f"Toplam istenen: {output['requested_count']}",
        "",
        "EN GUCLU 10 LONG",
    ]

    if long_ranking:
        for index, item in enumerate(long_ranking, start=1):
            lines.append(
                f"{index}. {item['symbol']} - "
                f"{item['long_score']}/100 "
                f"({item['long_label']})"
            )
    else:
        lines.append("Analiz sonucu yok.")

    lines.extend([
        "",
        "EN GUCLU 10 SHORT",
    ])

    if short_ranking:
        for index, item in enumerate(short_ranking, start=1):
            lines.append(
                f"{index}. {item['symbol']} - "
                f"{item['short_score']}/100 "
                f"({item['short_label']})"
            )
    else:
        lines.append("Analiz sonucu yok.")

    lines.extend([
        "",
        "Not: Puanlar teknik tarama sonucudur, "
        "tek basina islem tavsiyesi degildir.",
    ])

    return "\n".join(lines)


def run_scan():
    print("=" * 60)
    print("FUTURE RADAR 50 - COKLU SOZLESME TARAMASI")
    print("=" * 60)

    symbols, assets = load_symbols()

    asset_info = {
        (item.get("symbol") or item.get("instId")): item
        for item in assets
    }

    print(f"\nPiyasadaki sozlesme sayisi: {len(assets)}")
    print(f"Taranacak sozlesme: {len(symbols)}")

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
    print(f"Basarili analiz: {len(results)}")
    print(f"Basarisiz analiz: {len(errors)}")
    print(f"Sonuc dosyasi: {OUTPUT_FILE}")

    print("\nEN YUKSEK LONG PUANLARI")
    for item in long_ranking[:TOP_COUNT]:
        print(item["symbol"], item["long_score"])

    print("\nEN YUKSEK SHORT PUANLARI")
    for item in short_ranking[:TOP_COUNT]:
        print(item["symbol"], item["short_score"])

    report = create_telegram_report(output)

    try:
        send_telegram_message(report)
    except Exception as error:
        print(f"TELEGRAM GONDERIM HATASI: {error}")
        raise


if __name__ == "__main__":
    run_scan()
