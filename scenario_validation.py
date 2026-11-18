
from scenario_validation import make_scenarios
from indicator_engine import calculate_indicators

for name, data in make_scenarios().items():
    result = calculate_indicators(data)
    signals = result["signals"]

    long_items = sorted(
        [(k, v) for k, v in signals.items() if v is not None and v > 0],
        key=lambda x: x[1],
        reverse=True
    )[:10]

    short_items = sorted(
        [(k, v) for k, v in signals.items() if v is not None and v < 0],
        key=lambda x: x[1]
    )[:10]

    print("\n" + "=" * 55)
    print(name)
    print("LONG:", result["long_score"], "SHORT:", result["short_score"])
    print("CHOP:", result["indicator_values"]["Choppiness_Index"])
    print("ADX:", result["indicator_values"]["ADX"])
    print("\nLONG KATKILARI")
    for k, v in long_items:
        print(k, round(v, 3))

    print("\nSHORT KATKILARI")
    for k, v in short_items:
        print(k, round(v, 3))
