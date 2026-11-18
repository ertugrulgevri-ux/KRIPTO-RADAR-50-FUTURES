
import numpy as np
import pandas as pd

from indicator_engine import calculate_indicators


CANDLE_COUNT = 300
SEED = 42


def create_ohlcv(close, volume):
    """Create valid OHLCV candles from a close-price series."""
    rng = np.random.default_rng(SEED)

    close = np.asarray(close, dtype=float)
    volume = np.asarray(volume, dtype=float)

    open_price = np.empty(len(close))
    open_price[0] = close[0] * 0.999

    open_price[1:] = close[:-1]

    spread = np.maximum(close * 0.0015, 0.000001)

    high = np.maximum(open_price, close) + spread * rng.uniform(
        0.5, 1.5, len(close)
    )

    low = np.minimum(open_price, close) - spread * rng.uniform(
        0.5, 1.5, len(close)
    )

    return pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def make_scenarios():
    rng = np.random.default_rng(SEED)
    x = np.arange(CANDLE_COUNT)

    # 1. SIDEWAYS MARKET
    sideways_close = (
        100
        + 0.7 * np.sin(x / 5)
        + 0.25 * np.sin(x / 13)
        + rng.normal(0, 0.035, CANDLE_COUNT)
    )

    sideways_volume = rng.uniform(900, 1100, CANDLE_COUNT)

    # 2. BULLISH PREPARATION
    bullish_base = (
        100
        + 0.003 * x
        + 0.000025 * (x ** 2)
        + 0.12 * np.sin(x / 7)
    )

    bullish_close = bullish_base + rng.normal(
        0, 0.025, CANDLE_COUNT
    )

    # Final candles accelerate upward.
    bullish_close[-20:] += np.linspace(
        0, 2.0, 20
    )

    bullish_volume = rng.uniform(
        850, 1100, CANDLE_COUNT
    )

    bullish_volume[-20:] *= np.linspace(
        1.2, 2.0, 20
    )

    # 3. BEARISH PREPARATION
    bearish_base = (
        120
        - 0.003 * x
        - 0.000025 * (x ** 2)
        + 0.12 * np.sin(x / 7)
    )

    bearish_close = bearish_base + rng.normal(
        0, 0.025, CANDLE_COUNT
    )

    bearish_close[-20:] -= np.linspace(
        0, 2.0, 20
    )

    bearish_volume = rng.uniform(
        850, 1100, CANDLE_COUNT
    )

    bearish_volume[-20:] *= np.linspace(
        1.2, 2.0, 20
    )

    return {
        "YATAY PIYASA": create_ohlcv(
            sideways_close, sideways_volume
        ),
        "YUKSELIS HAZIRLIGI": create_ohlcv(
            bullish_close, bullish_volume
        ),
        "DUSUS HAZIRLIGI": create_ohlcv(
            bearish_close, bearish_volume
        ),
    }


def run_tests():
    print("=" * 65)
    print("FUTURE RADAR 50 - THREE SCENARIO VALIDATION")
    print("=" * 65)

    scenarios = make_scenarios()
    results = {}

    for name, data in scenarios.items():
        print("\n" + "-" * 65)
        print(name)
        print("-" * 65)

        result = calculate_indicators(data)
        results[name] = result

        print(f"LONG PUANI: {result['long_score']}")
        print(f"SHORT PUANI: {result['short_score']}")
        print(f"PIYASA REJIMI: {result['market_regime']}")
        print(
            "TREND CARPANI: "
            f"{result['trend_signal_multiplier']}"
        )
        print(
            "YATAYLIK SEVIYESI: "
            f"{result['sideways_strength']}"
        )
        print(
            "GECERLI FAKTOR: "
            f"{result['valid_signal_count']}/54"
        )

        print("\nOZEL FAKTORLER:")
        for factor, value in result["special_values"].items():
            print(f"{factor}: {value}")

    print("\n" + "=" * 65)
    print("SENARYO KARSILASTIRMASI")
    print("=" * 65)

    print(
        f"{'SENARYO':25}"
        f"{'LONG':>10}"
        f"{'SHORT':>10}"
        f"{'FARK':>10}"
    )

    for name, result in results.items():
        difference = round(
            result["long_score"] - result["short_score"], 2
        )

        print(
            f"{name:25}"
            f"{result['long_score']:>10.2f}"
            f"{result['short_score']:>10.2f}"
            f"{difference:>10.2f}"
        )

    print("\n" + "=" * 65)
    print("TEST TAMAMLANDI")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
