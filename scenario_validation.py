
import numpy as np
import pandas as pd

from indicator_engine import calculate_indicators


CANDLE_COUNT = 300
SEED = 42

SPECIAL_FACTORS = {
    "Hidden_Accumulation",
    "Rising_Lows_Below_Resistance",
    "Resistance_Absorption",
    "Volume_Breakout_Preparation",
}


def create_ohlcv(close, volume):
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

    # YATAY PIYASA
    sideways_close = (
        100
        + 0.7 * np.sin(x / 5)
        + 0.25 * np.sin(x / 13)
        + rng.normal(0, 0.035, CANDLE_COUNT)
    )

    sideways_volume = rng.uniform(900, 1100, CANDLE_COUNT)

    # YUKSELIS HAZIRLIGI
    bullish_base = (
        100
        + 0.003 * x
        + 0.000025 * (x ** 2)
        + 0.12 * np.sin(x / 7)
    )

    bullish_close = bullish_base + rng.normal(
        0, 0.025, CANDLE_COUNT
    )

    bullish_close[-20:] += np.linspace(0, 2.0, 20)

    bullish_volume = rng.uniform(
        850, 1100, CANDLE_COUNT
    )

    bullish_volume[-20:] *= np.linspace(1.2, 2.0, 20)

    # DUSUS HAZIRLIGI
    bearish_base = (
        120
        - 0.003 * x
        - 0.000025 * (x ** 2)
        + 0.12 * np.sin(x / 7)
    )

    bearish_close = bearish_base + rng.normal(
        0, 0.025, CANDLE_COUNT
    )

    bearish_close[-20:] -= np.linspace(0, 2.0, 20)

    bearish_volume = rng.uniform(
        850, 1100, CANDLE_COUNT
    )

    bearish_volume[-20:] *= np.linspace(1.2, 2.0, 20)

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


def print_top_factors(signals, score_weight_units, direction):
    factors = []

    for name, signal in signals.items():
        if signal is None:
            continue

        weight = 2 if name in SPECIAL_FACTORS else 1

        if direction == "LONG" and signal > 0:
            contribution = signal * weight
        elif direction == "SHORT" and signal < 0:
            contribution = abs(signal) * weight
        else:
            continue

        score_points = (
            100 * contribution / score_weight_units
            if score_weight_units
            else 0
        )

        factors.append((name, signal, weight, score_points))

    factors.sort(key=lambda item: item[3], reverse=True)

    for name, signal, weight, points in factors[:10]:
        print(
            f"{name:35} "
            f"sinyal={signal:7.3f} "
            f"agirlik={weight} "
            f"puan_katkisi={points:7.3f}"
        )


def report_scenario(name, data):
    result = calculate_indicators(data)

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(f"LONG PUANI: {result['long_score']}")
    print(f"SHORT PUANI: {result['short_score']}")
    print(f"PIYASA REJIMI: {result['market_regime']}")
    print(
        "CHOPPINESS INDEX: "
        f"{result['indicator_values'].get('Choppiness_Index')}"
    )
    print(
        "ADX: "
        f"{result['indicator_values'].get('ADX')}"
    )
    print(
        "TREND CARPANI: "
        f"{result['trend_signal_multiplier']}"
    )
    print(
        "YATAYLIK SEVIYESI: "
        f"{result['sideways_strength']}"
    )
    print(f"GECERLI FAKTOR: {result['valid_signal_count']}/54")

    print("\nLONG PUANINA EN COK KATKI VEREN 10 FAKTOR")
    print_top_factors(
        result["signals"],
        result["score_weight_units"],
        "LONG",
    )

    print("\nSHORT PUANINA EN COK KATKI VEREN 10 FAKTOR")
    print_top_factors(
        result["signals"],
        result["score_weight_units"],
        "SHORT",
    )

    print("\nOZEL FAKTORLER")
    for factor, value in result["special_values"].items():
        print(f"{factor:35} {value}")


def run_tests():
    print("=" * 70)
    print("FUTURE RADAR 50 - FACTOR CONTRIBUTION ANALYSIS")
    print("=" * 70)

    scenarios = make_scenarios()

    for name, data in scenarios.items():
        report_scenario(name, data)

    print("\n" + "=" * 70)
    print("ANALIZ TAMAMLANDI")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
