
import numpy as np
import pandas as pd


# =========================================================
# FUTURE RADAR 50
# 50 TECHNICAL INDICATORS + 4 SPECIAL FACTORS
# Pure Python / Pandas / NumPy
# =========================================================

TECHNICAL_WEIGHT = 1.0
SPECIAL_WEIGHT = 2.0

TIMEFRAME_WEIGHTS = {
    "5m": 0.10,
    "15m": 0.20,
    "1h": 0.30,
    "4h": 0.25,
    "1d": 0.15,
}


def _safe_div(a, b):
    return a / b.replace(0, np.nan)


def _clip(series):
    return pd.Series(series).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0).clip(-1, 1)


def _direction(value, threshold=0):
    return _clip(np.sign(value - threshold))


def calculate_indicators(data):
    """
    Input columns:
    open, high, low, close, volume

    Returns:
    indicator_values: 50 technical indicator values
    special_values: 4 special factor values
    signals: directional scores for all 54 factors
    """

    df = data.copy()
    df.columns = [str(c).lower() for c in df.columns]

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).reset_index(drop=True)

    if len(df) < 60:
        raise ValueError(
            "At least 60 candles are required."
        )

    o, h, l, c, v = (
        df["open"], df["high"], df["low"],
        df["close"], df["volume"]
    )

    prev_c = c.shift(1)
    change = c.pct_change()

    # ---------------- TREND ----------------

    sma = c.rolling(20).mean()
    ema = c.ewm(span=20, adjust=False).mean()

    wma_weights = np.arange(1, 21)
    wma = c.rolling(20).apply(
        lambda x: np.dot(x, wma_weights) /
        wma_weights.sum(), raw=True
    )

    half = c.rolling(10).mean()
    full = c.rolling(20).mean()
    hma = 2 * half - full

    vwma = _safe_div(
        (c * v).rolling(20).sum(),
        v.rolling(20).sum()
    )

    tenkan = (
        h.rolling(9).max() + l.rolling(9).min()
    ) / 2

    kijun = (
        h.rolling(26).max() + l.rolling(26).min()
    ) / 2

    span_a = (tenkan + kijun) / 2
    span_b = (
        h.rolling(52).max() + l.rolling(52).min()
    ) / 2

    ichimoku = (
        (c > span_a) & (c > span_b) & (tenkan > kijun)
    ).astype(float) - (
        (c < span_a) & (c < span_b) & (tenkan < kijun)
    ).astype(float)

    tr = pd.concat([
        h - l,
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(14).mean()

    mid = (h + l) / 2
    super_upper = mid + 3 * atr
    super_lower = mid - 3 * atr

    supertrend = pd.Series(
        np.where(c >= super_lower, 1, -1), index=df.index
    )

    psar = pd.Series(
        np.where(c >= c.shift(5), 1, -1), index=df.index
    )

    up_move = h.diff()
    down_move = -l.diff()

    aroon_up = (
        h.rolling(25).apply(
            lambda x: np.argmax(x), raw=True
        ) / 24 * 100
    )

    aroon_down = (
        l.rolling(25).apply(
            lambda x: np.argmin(x), raw=True
        ) / 24 * 100
    )

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move, 0
        ), index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move, 0
        ), index=df.index
    )

    plus_di = 100 * _safe_div(
        plus_dm.rolling(14).mean(), atr
    )

    minus_di = 100 * _safe_div(
        minus_dm.rolling(14).mean(), atr
    )

    vortex_plus = (
        (h - l.shift(1)).abs().rolling(14).sum()
    ) / tr.rolling(14).sum()

    vortex_minus = (
        (l - h.shift(1)).abs().rolling(14).sum()
    ) / tr.rolling(14).sum()

    # ---------------- MOMENTUM ----------------

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = _safe_div(gain, loss)
    rsi = 100 - (100 / (1 + rs))

    lowest = l.rolling(14).min()
    highest = h.rolling(14).max()

    stoch = 100 * _safe_div(
        c - lowest, highest - lowest
    )

    rsi_low = rsi.rolling(14).min()
    rsi_high = rsi.rolling(14).max()

    stoch_rsi = 100 * _safe_div(
        rsi - rsi_low, rsi_high - rsi_low
    )

    cci_tp = (h + l + c) / 3
    cci_mean = cci_tp.rolling(20).mean()
    cci_dev = cci_tp.rolling(20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True
    )
    cci = _safe_div(
        cci_tp - cci_mean, 0.015 * cci_dev
    )

    williams = -100 * _safe_div(
        highest - c, highest - lowest
    )

    typical = (h + l + c) / 3
    raw_money = typical * v

    positive_money = raw_money.where(
        typical.diff() > 0, 0
    )
    negative_money = raw_money.where(
        typical.diff() < 0, 0
    )

    money_ratio = _safe_div(
        positive_money.rolling(14).sum(),
        negative_money.rolling(14).sum()
    )

    mfi = 100 - 100 / (1 + money_ratio)

    roc = c.pct_change(12) * 100
    momentum = c - c.shift(10)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    ppo = 100 * _safe_div(
        ema12 - ema26, ema26
    )

    trix_ema = c.ewm(span=15, adjust=False).mean()
    trix_ema = trix_ema.ewm(span=15, adjust=False).mean()
    trix_ema = trix_ema.ewm(span=15, adjust=False).mean()
    trix = trix_ema.pct_change() * 100

    bp = c - pd.concat([l, prev_c], axis=1).min(axis=1)
    tr_uo = pd.concat([
        h - l, (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)

    avg7 = _safe_div(bp.rolling(7).sum(), tr_uo.rolling(7).sum())
    avg14 = _safe_div(bp.rolling(14).sum(), tr_uo.rolling(14).sum())
    avg28 = _safe_div(bp.rolling(28).sum(), tr_uo.rolling(28).sum())

    ultimate = 100 * (
        4 * avg7 + 2 * avg14 + avg28
    ) / 7

    pc = c.diff()
    abs_pc = pc.abs()
    tsi = 100 * _safe_div(
        pc.ewm(span=25).mean().ewm(span=13).mean(),
        abs_pc.ewm(span=25).mean().ewm(span=13).mean()
    )

    dpo = c.shift(11) - c.rolling(20).mean()

    # ---------------- TREND STRENGTH ----------------

    dx = 100 * _safe_div(
        (plus_di - minus_di).abs(),
        plus_di + minus_di
    )

    adx = dx.rolling(14).mean()

    sum_tr = tr.rolling(14).sum()
    high14 = h.rolling(14).max()
    low14 = l.rolling(14).min()

    chop = 100 * np.log10(
        _safe_div(sum_tr, high14 - low14)
    ) / np.log10(14)

    rvi_num = (
        (c - o) + 2 * (c.shift(1) - o.shift(1))
        + 2 * (c.shift(2) - o.shift(2))
        + (c.shift(3) - o.shift(3))
    ) / 6

    rvi_den = (
        (h - l) + 2 * (h.shift(1) - l.shift(1))
        + 2 * (h.shift(2) - l.shift(2))
        + (h.shift(3) - l.shift(3))
    ) / 6

    rvi = _safe_div(
        rvi_num.rolling(10).mean(),
        rvi_den.rolling(10).mean()
    )

    stc = macd.ewm(span=23).mean()

    # ---------------- VOLATILITY ----------------

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()

    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    kc_mid = c.ewm(span=20).mean()
    kc_upper = kc_mid + 2 * atr
    kc_lower = kc_mid - 2 * atr

    donchian_high = h.rolling(20).max()
    donchian_low = l.rolling(20).min()

    stddev = bb_std
    hist_vol = change.rolling(20).std() * np.sqrt(365) * 100

    drawdown = c / c.rolling(14).max() - 1
    ulcer = np.sqrt(
        (drawdown.pow(2)).rolling(14).mean()
    )

    chaikin_vol = (
        (h - l).ewm(span=10).mean().pct_change(10) * 100
    )

    # ---------------- VOLUME ----------------

    direction = np.sign(c.diff()).fillna(0)

    obv = (direction * v).cumsum()

    money_flow_multiplier = _safe_div(
        (c - l) - (h - c), h - l
    ).fillna(0)

    money_flow_volume = money_flow_multiplier * v

    cmf = _safe_div(
        money_flow_volume.rolling(20).sum(),
        v.rolling(20).sum()
    )

    adl = money_flow_volume.cumsum()

    force_index = (c.diff() * v).ewm(span=13).mean()

    eom = _safe_div(
        ((h + l) / 2).diff() * (h - l),
        v
    ).rolling(14).mean()

    volume_roc = v.pct_change(10) * 100

    relative_volume = _safe_div(
        v, v.rolling(20).mean()
    )

    vwap = _safe_div(
        (typical * v).rolling(20).sum(),
        v.rolling(20).sum()
    )

    pvt = (change.fillna(0) * v).cumsum()

    nvi = pd.Series(1000.0, index=df.index)
    pvi = pd.Series(1000.0, index=df.index)

    for i in range(1, len(df)):
        nvi.iloc[i] = (
            nvi.iloc[i - 1] * (1 + change.iloc[i])
            if v.iloc[i] < v.iloc[i - 1]
            else nvi.iloc[i - 1]
        )
        pvi.iloc[i] = (
            pvi.iloc[i - 1] * (1 + change.iloc[i])
            if v.iloc[i] > v.iloc[i - 1]
            else pvi.iloc[i - 1]
        )

    klinger_volume = (
        (2 * _safe_div(c - l, h - l) - 1) * v
    )
    klinger = (
        klinger_volume.ewm(span=34).mean()
        - klinger_volume.ewm(span=55).mean()
    )

    # ---------------- 50 INDICATOR VALUES ----------------

    values = {
        "SMA": sma,
        "EMA": ema,
        "WMA": wma,
        "HMA": hma,
        "VWMA": vwma,
        "Ichimoku": ichimoku,
        "Supertrend": supertrend,
        "Parabolic_SAR": psar,
        "Aroon": aroon_up - aroon_down,
        "Vortex": vortex_plus - vortex_minus,
        "RSI": rsi,
        "Stochastic": stoch,
        "Stochastic_RSI": stoch_rsi,
        "CCI": cci,
        "Williams_R": williams,
        "MFI": mfi,
        "ROC": roc,
        "Momentum": momentum,
        "MACD": macd - macd_signal,
        "PPO": ppo,
        "TRIX": trix,
        "Ultimate_Oscillator": ultimate,
        "TSI": tsi,
        "DPO": dpo,
        "ADX": adx,
        "Plus_DI": plus_di,
        "Minus_DI": minus_di,
        "Choppiness_Index": chop,
        "Relative_Vigor_Index": rvi,
        "Schaff_Trend_Cycle": stc,
        "ATR": atr,
        "Bollinger_Bands": bb_upper - bb_lower,
        "Keltner_Channels": kc_upper - kc_lower,
        "Donchian_Channels": donchian_high - donchian_low,
        "Standard_Deviation": stddev,
        "Historical_Volatility": hist_vol,
        "Ulcer_Index": ulcer,
        "Chaikin_Volatility": chaikin_vol,
        "OBV": obv,
        "Chaikin_Money_Flow": cmf,
        "Accumulation_Distribution": adl,
        "Force_Index": force_index,
        "Ease_of_Movement": eom,
        "Volume_ROC": volume_roc,
        "Relative_Volume": relative_volume,
        "VWAP": vwap,
        "Price_Volume_Trend": pvt,
        "Negative_Volume_Index": nvi,
        "Positive_Volume_Index": pvi,
        "Klinger_Oscillator": klinger,
    }

    # ---------------- DIRECTIONAL SIGNALS ----------------

    signals = {}

    signals["SMA"] = _direction(c - sma)
    signals["EMA"] = _direction(c - ema)
    signals["WMA"] = _direction(c - wma)
    signals["HMA"] = _direction(c - hma)
    signals["VWMA"] = _direction(c - vwma)
    signals["Ichimoku"] = _clip(ichimoku)
    signals["Supertrend"] = _clip(supertrend)
    signals["Parabolic_SAR"] = _clip(psar)
    signals["Aroon"] = _direction(aroon_up - aroon_down)
    signals["Vortex"] = _direction(vortex_plus - vortex_minus)

    signals["RSI"] = _clip((rsi - 50) / 25)
    signals["Stochastic"] = _clip((stoch - 50) / 50)
    signals["Stochastic_RSI"] = _clip((stoch_rsi - 50) / 50)
    signals["CCI"] = _clip(cci / 150)
    signals["Williams_R"] = _clip((williams + 50) / 50)
    signals["MFI"] = _clip((mfi - 50) / 25)
    signals["ROC"] = _clip(roc / 5)
    signals["Momentum"] = _direction(momentum)
    signals["MACD"] = _direction(macd - macd_signal)
    signals["PPO"] = _direction(ppo)
    signals["TRIX"] = _direction(trix)
    signals["Ultimate_Oscillator"] = _clip((ultimate - 50) / 25)
    signals["TSI"] = _clip(tsi / 50)
    signals["DPO"] = _direction(dpo)

    signals["ADX"] = _clip((adx - 20) / 25) * _direction(plus_di - minus_di)
    signals["Plus_DI"] = _direction(plus_di - minus_di)
    signals["Minus_DI"] = _direction(minus_di - plus_di)
    signals["Choppiness_Index"] = _clip((50 - chop) / 30) * _direction(c - sma)
    signals["Relative_Vigor_Index"] = _direction(rvi)
    signals["Schaff_Trend_Cycle"] = _direction(stc - stc.shift(1))

    signals["ATR"] = _direction(c - prev_c)
    signals["Bollinger_Bands"] = _clip(
        _safe_div(c - bb_mid, 2 * bb_std)
    )
    signals["Keltner_Channels"] = _direction(c - kc_mid)
    signals["Donchian_Channels"] = _direction(c - (donchian_high + donchian_low) / 2)
    signals["Standard_Deviation"] = _direction(c - prev_c)
    signals["Historical_Volatility"] = _direction(c - prev_c)
    signals["Ulcer_Index"] = _direction(c - sma)
    signals["Chaikin_Volatility"] = _direction(c - prev_c)

    signals["OBV"] = _direction(obv - obv.shift(10))
    signals["Chaikin_Money_Flow"] = _clip(cmf * 3)
    signals["Accumulation_Distribution"] = _direction(adl - adl.shift(10))
    signals["Force_Index"] = _direction(force_index)
    signals["Ease_of_Movement"] = _direction(eom)
    signals["Volume_ROC"] = _direction(volume_roc)
    signals["Relative_Volume"] = _direction(relative_volume - 1)
    signals["VWAP"] = _direction(c - vwap)
    signals["Price_Volume_Trend"] = _direction(pvt - pvt.shift(10))
    signals["Negative_Volume_Index"] = _direction(nvi - nvi.shift(10))
    signals["Positive_Volume_Index"] = _direction(pvi - pvi.shift(10))
    signals["Klinger_Oscillator"] = _direction(klinger)

    # ---------------- 4 SPECIAL FACTORS ----------------

    resistance = h.shift(1).rolling(20).max()
    support = l.shift(1).rolling(20).min()

    higher_lows = (
        (l > l.shift(5)) & (l.shift(5) > l.shift(10))
    )

    near_resistance = _safe_div(
        resistance - c, c
    ).abs() < 0.015

    rising_lows_factor = pd.Series(
        np.where(higher_lows & near_resistance, 1, 0),
        index=df.index
    )

    # Approximation using candle body, wick and volume.
    # True order-book absorption requires order-book/trade data.
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    candle_range = (h - l).replace(0, np.nan)

    absorption_factor = (
        (v > v.rolling(20).mean() * 1.3)
        & (upper_wick / candle_range < 0.35)
        & (c >= c.shift(1))
    ).astype(float)

    # Hidden accumulation: rising OBV while price stays relatively flat.
    price_flat = c.pct_change(10).abs() < 0.03
    obv_rising = obv > obv.shift(10)

    accumulation_factor = (
        price_flat & obv_rising
    ).astype(float)

    # Volume and breakout preparation.
    volume_expanding = v > v.rolling(20).mean() * 1.5
    approaching_resistance = c >= resistance * 0.98

    breakout_factor = (
        volume_expanding & approaching_resistance
    ).astype(float)

    special = {
        "Hidden_Accumulation": accumulation_factor,
        "Rising_Lows_Below_Resistance": rising_lows_factor,
        "Resistance_Absorption": absorption_factor,
        "Volume_Breakout_Preparation": breakout_factor,
    }

    # Special factors are directional: positive supports LONG.
    for name, series in special.items():
        signals[name] = _clip(series)

    # ---------------- FINAL SCORES ----------------

    latest = {
        name: float(_clip(series).iloc[-1])
        for name, series in signals.items()
    }

    long_total = 0.0
    short_total = 0.0
    total_weight = 0.0

    for name, value in latest.items():
        weight = (
            SPECIAL_WEIGHT
            if name in special
            else TECHNICAL_WEIGHT
        )

        long_total += max(value, 0) * weight
        short_total += max(-value, 0) * weight
        total_weight += weight

    long_score = round(100 * long_total / total_weight, 2)
    short_score = round(100 * short_total / total_weight, 2)

    return {
        "indicator_count": len(values),
        "special_factor_count": len(special),
        "total_factor_count": len(latest),
        "indicator_values": {
            k: float(v.iloc[-1])
            if pd.notna(v.iloc[-1]) else None
            for k, v in values.items()
        },
        "special_values": {
            k: float(v.iloc[-1])
            for k, v in special.items()
        },
        "signals": latest,
        "long_score": long_score,
        "short_score": short_score,
        "score_weight_units": total_weight,
    }


def score_label(score):
    if score >= 90:
        return "OLAGANUSTU SINYAL"
    if score >= 80:
        return "COK GUCLU HAZIRLIK"
    if score >= 65:
        return "GUCLU HAZIRLIK"
    if score >= 50:
        return "TAKIP"
    if score >= 30:
        return "DUSUK"
    return "ZAYIF"


def analyze_timeframes(timeframe_data):
    """
    timeframe_data example:
    {
        "5m": dataframe,
        "15m": dataframe,
        "1h": dataframe,
        "4h": dataframe,
        "1d": dataframe
    }
    """

    results = {}
    long_score = 0.0
    short_score = 0.0
    available_weight = 0.0

    for timeframe, weight in TIMEFRAME_WEIGHTS.items():
        if timeframe not in timeframe_data:
            continue

        result = calculate_indicators(
            timeframe_data[timeframe]
        )

        results[timeframe] = result
        long_score += result["long_score"] * weight
        short_score += result["short_score"] * weight
        available_weight += weight

    if available_weight == 0:
        raise ValueError("No timeframe data supplied.")

    # Missing timeframes are not silently treated as zero.
    final_long = round(long_score / available_weight, 2)
    final_short = round(short_score / available_weight, 2)

    return {
        "timeframes": results,
        "long_score": final_long,
        "short_score": final_short,
        "long_label": score_label(final_long),
        "short_label": score_label(final_short),
        "available_timeframe_weight": round(
            available_weight, 4
        ),
        "complete_timeframes": (
            len(results) == len(TIMEFRAME_WEIGHTS)
        ),
    }
