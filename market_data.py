
import time
import requests
import pandas as pd


BASE_URL = "https://www.okx.com"
CANDLE_LIMIT = 300
MIN_CANDLES = 120

TIMEFRAMES = {
    "5m": "5m",
    "15m": "15m",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
}


def fetch_candles(
    inst_id,
    timeframe,
    limit=CANDLE_LIMIT,
    min_candles=MIN_CANDLES
):
    """
    OKX'ten kapanmis mumlari alir.
    Sonucu indicator_engine icin DataFrame olarak dondurur.
    """

    if timeframe not in TIMEFRAMES:
        raise ValueError(
            f"Gecersiz zaman dilimi: {timeframe}"
        )

    if not 1 <= limit <= 300:
        raise ValueError("Mum sayisi 1-300 arasinda olmali.")

    url = f"{BASE_URL}/api/v5/market/candles"

    params = {
        "instId": inst_id,
        "bar": TIMEFRAMES[timeframe],
        "limit": str(limit),
    }

    last_error = None

    for attempt in range(3):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=20
            )

            response.raise_for_status()
            result = response.json()

            if result.get("code") != "0":
                raise RuntimeError(
                    f"OKX API hatasi: "
                    f"{result.get('code')} "
                    f"{result.get('msg')}"
                )

            raw_candles = result.get("data", [])

            if not raw_candles:
                raise RuntimeError(
                    f"{inst_id} {timeframe}: Mum verisi bos."
                )

            rows = []

            for candle in raw_candles:

                if len(candle) < 9:
                    continue

                # Yalnizca kapanmis mumlar
                if candle[8] != "1":
                    continue

                rows.append({
                    "timestamp": int(candle[0]),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),

                    # OKX volCcy: SWAP icin baz para hacmi
                    "volume": float(candle[6]),
                })

            if not rows:
                raise RuntimeError(
                    f"{inst_id} {timeframe}: "
                    "Kapanmis mum bulunamadi."
                )

            df = pd.DataFrame(rows)

            # Eskiden yeniye sirala
            df = df.sort_values(
                "timestamp"
            ).drop_duplicates(
                subset=["timestamp"],
                keep="last"
            ).reset_index(drop=True)

            # Zaman damgasini UTC tarihine cevir
            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                unit="ms",
                utc=True
            )

            # OHLC verisini kontrol et
            if (
                (df["high"] < df["low"]).any()
                or (df["volume"] < 0).any()
                or (df["open"] <= 0).any()
                or (df["high"] <= 0).any()
                or (df["low"] <= 0).any()
                or (df["close"] <= 0).any()
            ):
                raise RuntimeError(
                    f"{inst_id} {timeframe}: "
                    "Gecersiz mum verisi."
                )

            if len(df) < min_candles:
                raise RuntimeError(
                    f"{inst_id} {timeframe}: "
                    f"Yalnizca {len(df)} kapanmis mum var. "
                    f"En az {min_candles} gerekli."
                )

            return df

        except Exception as error:
            last_error = error

            if attempt < 2:
                time.sleep(2 * (attempt + 1))

    raise RuntimeError(
        f"{inst_id} {timeframe} veri alma basarisiz: "
        f"{last_error}"
    )


def fetch_all_timeframes(inst_id):
    """
    Bir sozlesmenin bes zaman dilimini alir.
    """

    results = {}

    for timeframe in TIMEFRAMES:

        print(
            f"{inst_id} - {timeframe} verisi aliniyor..."
        )

        df = fetch_candles(
            inst_id,
            timeframe
        )

        results[timeframe] = df

        print(
            f"  Basarili: {len(df)} kapanmis mum"
        )

        time.sleep(0.15)

    return results


if __name__ == "__main__":

    print("=" * 55)
    print("FUTURE RADAR 50 - OKX MUM VERISI TESTI")
    print("=" * 55)

    symbol = "BTC-USDT-SWAP"

    market_data = fetch_all_timeframes(symbol)

    print("\nTEST SONUCU")
    print("-" * 55)

    for timeframe, df in market_data.items():

        print(
            f"{timeframe}: "
            f"{len(df)} mum | "
            f"Son kapanis: {df.iloc[-1]['close']}"
        )

    print("\nBES ZAMAN DILIMI BASARIYLA ALINDI.")
