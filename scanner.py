
from pprint import pprint

from market_data import fetch_all_timeframes
from indicator_engine import analyze_timeframes


SYMBOL = "BTC-USDT-SWAP"


def run_scan(symbol):

    print("=" * 60)
    print("FUTURE RADAR 50 - ILK CANLI ANALIZ")
    print("=" * 60)

    # 1. Bes zaman diliminin mumlarini al
    print("\n[1/2] Piyasa verileri aliniyor...")

    timeframe_data = fetch_all_timeframes(symbol)

    expected_timeframes = {
        "5m", "15m", "1h", "4h", "1d"
    }

    if set(timeframe_data.keys()) != expected_timeframes:
        raise RuntimeError(
            "Bes zaman diliminin tamami alinamadi."
        )

    print("\nTum zaman dilimleri hazir.")

    # 2. Gosterge motoruna gonder
    print("\n[2/2] Gosterge motoru calistiriliyor...")

    result = analyze_timeframes(timeframe_data)

    print("\n" + "=" * 60)
    print("ANALIZ SONUCU")
    print("=" * 60)

    print("\nSonuc veri tipi:", type(result).__name__)

    if isinstance(result, dict):

        print("\nSonuc alanlari:")

        for key in result:
            print("-", key)

    print("\nDETAYLI SONUC:")
    pprint(result, width=100, depth=4)

    print("\n" + "=" * 60)
    print("ILK CANLI ANALIZ TAMAMLANDI")
    print("=" * 60)

    return result


if __name__ == "__main__":
    run_scan(SYMBOL)
