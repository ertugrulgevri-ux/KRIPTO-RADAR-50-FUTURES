
import numpy as np
import pandas as pd

# FUTURE RADAR 50
# 50 technical indicators + 4 special early-warning factors.

TECHNICAL_WEIGHT = 1.0
SPECIAL_WEIGHT = 2.0

TIMEFRAME_WEIGHTS = {
    "5m": 0.10,
    "15m": 0.20,
    "1h": 0.30,
    "4h": 0.25,
    "1d": 0.15,
}

MIN_CANDLES = 120


def _safe_div(a, b):
    if np.isscalar(b):
        return a / (b if b != 0 else np.nan)

    return a / b.replace(0, np.nan)


def _clip(x, low=-1.0, high=1.0):
    if isinstance(x, pd.Series):
        return x.replace(
            [np.inf, -np.inf], np.nan
        ).clip(low, high)

    return float(np.clip(x, low, high)) if pd.notna(x) else np.nan


def _direction(x, scale=None):
    if scale is None:
        return _clip(np.sign(x))

    return _clip(x / scale)


def _wma(series, period):
    weights = np.arange(1, period + 1, dtype=float)

    return series.rolling(
        period,
        min_periods=period
    ).apply(
        lambda values: np.dot(values, weights) / weights.sum(),
        raw=True
    )


def _rma(series, period):
    """Wilder smoothing, initialized with an SMA."""

    out = pd.Series(
        np.nan,
        index=series.index,
        dtype=float
    )

    if len(series) < period:
        return out

    first = series.rolling(
        period,
        min_periods=period
    ).mean()

    valid = first.first_valid_index()

    if valid is None:
        return out

    pos = series.index.get_loc(valid)
    out.iloc[pos] = first.iloc[pos]

    for i in range(pos + 1, len(series)):
        previous = out.iloc[i - 1]
        current = series.iloc[i]

        if pd.notna(previous) and pd.notna(current):
            out.iloc[i] = (
                previous * (period - 1) + current
            ) / period

    return out


def _psar(high, low, step=0.02, max_step=0.20):
    n = len(high)

    sar = pd.Series(
        np.nan,
        index=high.index,
        dtype=float
    )

    trend = pd.Series(
        0.0,
        index=high.index
    )

    if n < 2:
        return sar, trend

    up = (
        high.iloc[1] + low.iloc[1]
    ) >= (
        high.iloc[0] + low.iloc[0]
    )

    af = step
    ep = high.iloc[0] if up else low.iloc[0]

    sar.iloc[0] = low.iloc[0] if up else high.iloc[0]
    trend.iloc[0] = 1.0 if up else -1.0

    for i in range(1, n):

        previous_sar = sar.iloc[i - 1]

        candidate = previous_sar + af * (
            ep - previous_sar
        )

        if up:

            candidate = min(
                candidate,
                low.iloc[i - 1],
                low.iloc[i - 2] if i >= 2 else low.iloc[i - 1]
            )

            if low.iloc[i] < candidate:

                up = False
                candidate = ep
                ep = low.iloc[i]
                af = step

            elif high.iloc[i] > ep:

                ep = high.iloc[i]
                af = min(af + step, max_step)

        else:

            candidate = max(
                candidate,
                high.iloc[i - 1],
                high.iloc[i - 2] if i >= 2 else high.iloc[i - 1]
            )

            if high.iloc[i] > candidate:

                up = True
                candidate = ep
                ep = high.iloc[i]
                af = step

            elif low.iloc[i] < ep:

                ep = low.iloc[i]
                af = min(af + step, max_step)

        sar.iloc[i] = candidate
        trend.iloc[i] = 1.0 if up else -1.0

    return sar, trend


def _supertrend(high, low, close, atr, multiplier=3.0):
    """Final-band Supertrend calculation."""

    midpoint = (high + low) / 2.0

    basic_upper = midpoint + multiplier * atr
    basic_lower = midpoint - multiplier * atr

    upper = pd.Series(np.nan, index=close.index)
    lower = pd.Series(np.nan, index=close.index)
    direction = pd.Series(np.nan, index=close.index)

    for i in range(len(close)):

        if pd.isna(atr.iloc[i]):
            continue

        if i == 0 or pd.isna(atr.iloc[i - 1]):

            upper.iloc[i] = basic_upper.iloc[i]
            lower.iloc[i] = basic_lower.iloc[i]
            direction.iloc[i] = 1.0
            continue

        previous_upper = upper.iloc[i - 1]
        previous_lower = lower.iloc[i - 1]

        if pd.isna(previous_upper) or pd.isna(previous_lower):

            upper.iloc[i] = basic_upper.iloc[i]
            lower.iloc[i] = basic_lower.iloc[i]
            direction.iloc[i] = 1.0
            continue

        if (
            basic_upper.iloc[i] < previous_upper
            or close.iloc[i - 1] > previous_upper
        ):
            upper.iloc[i] = basic_upper.iloc[i]
        else:
            upper.iloc[i] = previous_upper

        if (
            basic_lower.iloc[i] > previous_lower
            or close.iloc[i - 1] < previous_lower
        ):
            lower.iloc[i] = basic_lower.iloc[i]
        else:
            lower.iloc[i] = previous_lower

        if direction.iloc[i - 1] > 0:

            direction.iloc[i] = (
                -1.0
                if close.iloc[i] < lower.iloc[i]
                else 1.0
            )

        else:

            direction.iloc[i] = (
                1.0
                if close.iloc[i] > upper.iloc[i]
                else -1.0
            )

    return direction


def _prepare_data(data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    df = data.copy()

    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing OHLCV columns: {missing}"
        )

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove infinite and invalid OHLCV rows.
    df[required] = df[required].replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(subset=required)

    # If the caller supplies candle status, exclude unfinished candles.
    if "is_closed" in df.columns:

        closed = df["is_closed"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )

        df = df.loc[closed].copy()

    # Sort and remove duplicate timestamps.
    timestamp_column = None

    if "open_time" in df.columns:
        timestamp_column = "open_time"

    elif "timestamp" in df.columns:
        timestamp_column = "timestamp"

    if timestamp_column:
        df = df.sort_values(timestamp_column)

        df = df.drop_duplicates(
            subset=[timestamp_column],
            keep="last"
        )

    df = df.reset_index(drop=True)

    if len(df) < MIN_CANDLES:
        raise ValueError(
            f"At least {MIN_CANDLES} closed candles are required; "
            f"received {len(df)}"
        )

    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]
    v = df["volume"]

    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")

    if (h < df[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("Invalid candle: high is below OHLC values")

    if (l > df[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Invalid candle: low is above OHLC values")

    if (v < 0).any():
        raise ValueError("Volume cannot be negative")

    return df


def calculate_indicators(data):
    """
    Calculate 50 technical indicators and 4 special factors.

    Input:
        OHLCV DataFrame.

    If an is_closed column is supplied, unfinished candles are excluded.
    Otherwise, the caller must provide closed candles only.
    """

    df = _prepare_data(data)

    o, h, l, c, v = (
        df[column].astype(float)
        for column in [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    prev_c = c.shift(1)
    change = c.pct_change(fill_method=None)

    typical = (h + l + c) / 3.0

    tr = pd.concat(
        [
            h - l,
            (h - prev_c).abs(),
            (l - prev_c).abs()
        ],
        axis=1
    ).max(axis=1)

    atr = _rma(tr, 14)

    # ==========================================================
    # 1-10: TREND INDICATORS
    # ==========================================================

    sma = c.rolling(20).mean()

    ema = c.ewm(
        span=20,
        adjust=False,
        min_periods=20
    ).mean()

    wma = _wma(c, 20)

    hma = _wma(
        2 * _wma(c, 10) - _wma(c, 20),
        4
    )

    vwma = _safe_div(
        (c * v).rolling(20).sum(),
        v.rolling(20).sum()
    )

    tenkan = (
        h.rolling(9).max()
        + l.rolling(9).min()
    ) / 2

    kijun = (
        h.rolling(26).max()
        + l.rolling(26).min()
    ) / 2

    span_a = (tenkan + kijun) / 2

    span_b = (
        h.rolling(52).max()
        + l.rolling(52).min()
    ) / 2

    ichimoku = np.sign(
        (c - span_a)
        + (c - span_b)
        + (tenkan - kijun)
    )

    supertrend = _supertrend(
        h, l, c, atr
    )

    psar_value, psar_trend = _psar(h, l)

    aroon_up = h.rolling(25).apply(
        lambda x: 100.0 * np.argmax(x) / 24.0,
        raw=True
    )

    aroon_down = l.rolling(25).apply(
        lambda x: 100.0 * np.argmin(x) / 24.0,
        raw=True
    )

    up_move = h.diff()
    down_move = -l.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0
        ),
        index=df.index
    )

    plus_di = 100 * _safe_div(
        _rma(plus_dm, 14),
        atr
    )

    minus_di = 100 * _safe_div(
        _rma(minus_dm, 14),
        atr
    )

    vortex_plus = _safe_div(
        (h - l.shift(1)).abs().rolling(14).sum(),
        tr.rolling(14).sum()
    )

    vortex_minus = _safe_div(
        (l - h.shift(1)).abs().rolling(14).sum(),
        tr.rolling(14).sum()
    )

    # ==========================================================
    # 11-24: MOMENTUM INDICATORS
    # ==========================================================

    delta = c.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = _rma(gain, 14)
    avg_loss = _rma(loss, 14)

    rs = _safe_div(avg_gain, avg_loss)

    rsi = 100 - 100 / (1 + rs)

    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain > 0),
        100.0
    )

    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain == 0),
        50.0
    )

    lowest = l.rolling(14).min()
    highest = h.rolling(14).max()

    stoch = 100 * _safe_div(
        c - lowest,
        highest - lowest
    )

    stoch = stoch.mask(
        highest == lowest,
        50.0
    )

    rsi_low = rsi.rolling(14).min()
    rsi_high = rsi.rolling(14).max()

    stoch_rsi = 100 * _safe_div(
        rsi - rsi_low,
        rsi_high - rsi_low
    )

    stoch_rsi = stoch_rsi.mask(
        rsi_high == rsi_low,
        50.0
    )

    cci_mean = typical.rolling(20).mean()

    cci_dev = typical.rolling(20).apply(
        lambda x: np.mean(
            np.abs(x - np.mean(x))
        ),
        raw=True
    )

    cci = _safe_div(
        typical - cci_mean,
        0.015 * cci_dev
    )

    cci = cci.mask(
        cci_dev == 0,
        0.0
    )

    williams = -100 * _safe_div(
        highest - c,
        highest - lowest
    )

    raw_money = typical * v

    pos_money = raw_money.where(
        typical.diff() > 0,
        0.0
    )

    neg_money = raw_money.where(
        typical.diff() < 0,
        0.0
    )

    pos_sum = pos_money.rolling(14).sum()
    neg_sum = neg_money.rolling(14).sum()

    money_ratio = _safe_div(
        pos_sum,
        neg_sum
    )

    mfi = 100 - 100 / (1 + money_ratio)

    mfi = mfi.mask(
        (neg_sum == 0) & (pos_sum > 0),
        100.0
    )

    mfi = mfi.mask(
        (neg_sum == 0) & (pos_sum == 0),
        50.0
    )

    roc = c.pct_change(
        12,
        fill_method=None
    ) * 100

    momentum = c - c.shift(10)

    ema12 = c.ewm(
        span=12,
        adjust=False,
        min_periods=12
    ).mean()

    ema26 = c.ewm(
        span=26,
        adjust=False,
        min_periods=26
    ).mean()

    macd = ema12 - ema26

    macd_signal = macd.ewm(
        span=9,
        adjust=False,
        min_periods=9
    ).mean()

    ppo = 100 * _safe_div(
        macd,
        ema26
    )

    trix_ema = (
        c.ewm(span=15, adjust=False, min_periods=15).mean()
        .ewm(span=15, adjust=False, min_periods=15).mean()
        .ewm(span=15, adjust=False, min_periods=15).mean()
    )

    trix = trix_ema.pct_change(
        fill_method=None
    ) * 100

    bp = c - pd.concat(
        [l, prev_c],
        axis=1
    ).min(axis=1)

    tr_uo = pd.concat(
        [
            h - l,
            (h - prev_c).abs(),
            (l - prev_c).abs()
        ],
        axis=1
    ).max(axis=1)

    avg7 = _safe_div(
        bp.rolling(7).sum(),
        tr_uo.rolling(7).sum()
    )

    avg14 = _safe_div(
        bp.rolling(14).sum(),
        tr_uo.rolling(14).sum()
    )

    avg28 = _safe_div(
        bp.rolling(28).sum(),
        tr_uo.rolling(28).sum()
    )

    ultimate = 100 * (
        4 * avg7 + 2 * avg14 + avg28
    ) / 7

    pc = c.diff()

    tsi_num = (
        pc.ewm(span=25, adjust=False).mean()
        .ewm(span=13, adjust=False).mean()
    )

    tsi_den = (
        pc.abs().ewm(span=25, adjust=False).mean()
        .ewm(span=13, adjust=False).mean()
    )

    tsi = 100 * _safe_div(
        tsi_num,
        tsi_den
    )

    tsi = tsi.mask(
        tsi_den == 0,
        0.0
    )

    dpo = c.shift(11) - c.rolling(20).mean()

    # ==========================================================
    # 25-30: TREND STRENGTH AND OSCILLATORS
    # ==========================================================

    di_sum = plus_di + minus_di

    dx = 100 * _safe_div(
        (plus_di - minus_di).abs(),
        di_sum
    )

    dx = dx.mask(
        di_sum == 0,
        0.0
    )

    adx = _rma(dx, 14)

    chop_sum = tr.rolling(14).sum()

    chop_range = (
        h.rolling(14).max()
        - l.rolling(14).min()
    )

    chop_ratio = _safe_div(
        chop_sum,
        chop_range
    )

    chop = (
        100 * np.log10(chop_ratio)
        / np.log10(14)
    )

    rvi_num = (
        (c - o)
        + 2 * (c.shift(1) - o.shift(1))
        + 2 * (c.shift(2) - o.shift(2))
        + (c.shift(3) - o.shift(3))
    ) / 6

    rvi_den = (
        (h - l)
        + 2 * (h.shift(1) - l.shift(1))
        + 2 * (h.shift(2) - l.shift(2))
        + (h.shift(3) - l.shift(3))
    ) / 6

    rvi = _safe_div(
        rvi_num.rolling(10).mean(),
        rvi_den.rolling(10).mean()
    )

    stc_fast = macd.rolling(10).min()
    stc_slow = macd.rolling(10).max()

    stc_k = 100 * _safe_div(
        macd - stc_fast,
        stc_slow - stc_fast
    )

    stc_k = stc_k.mask(
        stc_slow == stc_fast,
        50.0
    )

    stc_d = stc_k.ewm(
        span=3,
        adjust=False
    ).mean()

    stc2_low = stc_d.rolling(10).min()
    stc2_high = stc_d.rolling(10).max()

    stc_raw = 100 * _safe_div(
        stc_d - stc2_low,
        stc2_high - stc2_low
    )

    stc_raw = stc_raw.mask(
        stc2_high == stc2_low,
        50.0
    )

    stc = stc_raw.ewm(
        span=3,
        adjust=False
    ).mean()

    # ==========================================================
    # 31-38: VOLATILITY INDICATORS
    # ==========================================================

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std(ddof=0)

    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    kc_mid = c.ewm(
        span=20,
        adjust=False,
        min_periods=20
    ).mean()

    kc_upper = kc_mid + 2 * atr
    kc_lower = kc_mid - 2 * atr

    donchian_high = h.rolling(20).max()
    donchian_low = l.rolling(20).min()

    hist_vol = (
        change.rolling(20).std(ddof=0)
        * np.sqrt(365)
        * 100
    )

    rolling_peak = c.rolling(14).max()

    drawdown_pct = (
        c / rolling_peak - 1
    ) * 100

    ulcer = np.sqrt(
        drawdown_pct.pow(2).rolling(14).mean()
    )

    chaikin_vol = (
        (h - l)
        .ewm(span=10, adjust=False)
        .mean()
        .pct_change(10, fill_method=None)
        * 100
    )

    # ==========================================================
    # 39-50: VOLUME AND MONEY FLOW
    # ==========================================================

    obv = (
        np.sign(c.diff()).fillna(0) * v
    ).cumsum()

    mfm = _safe_div(
        (c - l) - (h - c),
        h - l
    )

    mfv = mfm.fillna(0) * v

    cmf = _safe_div(
        mfv.rolling(20).sum(),
        v.rolling(20).sum()
    )

    adl = mfv.cumsum()

    force = (
        delta * v
    ).ewm(
        span=13,
        adjust=False,
        min_periods=13
    ).mean()

    midpoint_move = (
        (h + l) / 2
    ).diff()

    box_ratio = _safe_div(
        v,
        h - l
    )

    eom = _safe_div(
        midpoint_move,
        box_ratio
    ).rolling(14).mean()

    volume_roc = v.pct_change(
        10,
        fill_method=None
    ) * 100

    relative_volume = _safe_div(
        v,
        v.rolling(20).mean()
    )

    vwap = _safe_div(
        (typical * v).rolling(20).sum(),
        v.rolling(20).sum()
    )

    pvt = (
        change.fillna(0) * v
    ).cumsum()

    nvi = pd.Series(
        1000.0,
        index=df.index
    )

    pvi = pd.Series(
        1000.0,
        index=df.index
    )

    for i in range(1, len(df)):

        ret = (
            change.iloc[i]
            if pd.notna(change.iloc[i])
            else 0.0
        )

        nvi.iloc[i] = (
            nvi.iloc[i - 1] * (1 + ret)
            if v.iloc[i] < v.iloc[i - 1]
            else nvi.iloc[i - 1]
        )

        pvi.iloc[i] = (
            pvi.iloc[i - 1] * (1 + ret)
            if v.iloc[i] > v.iloc[i - 1]
            else pvi.iloc[i - 1]
        )

    trend = (
        np.sign((h + l + c).diff())
        .replace(0, np.nan)
        .ffill()
        .fillna(1.0)
    )

    dm = h - l

    cm = pd.Series(
        np.nan,
        index=df.index,
        dtype=float
    )

    for i in range(len(df)):

        if i == 0 or trend.iloc[i] != trend.iloc[i - 1]:

            cm.iloc[i] = (
                dm.iloc[i]
                + (dm.iloc[i - 1] if i > 0 else 0.0)
            )

        else:

            cm.iloc[i] = cm.iloc[i - 1] + dm.iloc[i]

    vf = (
        v
        * (2 * _safe_div(dm, cm).fillna(0) - 1)
        * trend
        * 100
    )

    klinger = (
        vf.ewm(span=34, adjust=False, min_periods=34).mean()
        - vf.ewm(span=55, adjust=False, min_periods=55).mean()
    )

    # ==========================================================
    # EXACTLY 50 TECHNICAL INDICATORS
    # ==========================================================

    values = {
        "SMA": sma,
        "EMA": ema,
        "WMA": wma,
        "HMA": hma,
        "VWMA": vwma,
        "Ichimoku": ichimoku,
        "Supertrend": supertrend,
        "Parabolic_SAR": psar_value,
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
        "Standard_Deviation": bb_std,
        "Historical_Volatility": hist_vol,
        "Ulcer_Index": ulcer,
        "Chaikin_Volatility": chaikin_vol,
        "OBV": obv,
        "Chaikin_Money_Flow": cmf,
        "Accumulation_Distribution": adl,
        "Force_Index": force,
        "Ease_of_Movement": eom,
        "Volume_ROC": volume_roc,
        "Relative_Volume": relative_volume,
        "VWAP": vwap,
        "Price_Volume_Trend": pvt,
        "Negative_Volume_Index": nvi,
        "Positive_Volume_Index": pvi,
        "Klinger_Oscillator": klinger,
    }

    # ==========================================================
    # DIRECTIONAL SIGNALS
    # ==========================================================

    signals = {
        "SMA": _direction(_safe_div(c - sma, atr)),
        "EMA": _direction(_safe_div(c - ema, atr)),
        "WMA": _direction(_safe_div(c - wma, atr)),
        "HMA": _direction(_safe_div(c - hma, atr)),
        "VWMA": _direction(_safe_div(c - vwma, atr)),

        "Ichimoku": _clip(ichimoku),
        "Supertrend": _clip(supertrend),
        "Parabolic_SAR": _clip(np.sign(c - psar_value)),

        "Aroon": _clip((aroon_up - aroon_down) / 100),
        "Vortex": _clip((vortex_plus - vortex_minus) / 0.5),

        "RSI": _clip((rsi - 50) / 25),
        "Stochastic": _clip((stoch - 50) / 50),
        "Stochastic_RSI": _clip((stoch_rsi - 50) / 50),
        "CCI": _clip(cci / 200),
        "Williams_R": _clip((williams + 50) / 50),
        "MFI": _clip((mfi - 50) / 25),

        "ROC": _clip(roc / 5),
        "Momentum": _direction(_safe_div(momentum, atr)),
        "MACD": _direction(_safe_div(macd - macd_signal, atr)),
        "PPO": _clip(ppo / 2),
        "TRIX": _clip(trix / 0.5),
        "Ultimate_Oscillator": _clip((ultimate - 50) / 25),
        "TSI": _clip(tsi / 50),
        "DPO": _direction(_safe_div(dpo, atr)),

        "ADX": (
            _clip((adx - 20) / 25)
            * _clip((plus_di - minus_di) / 25)
        ),

        "Plus_DI": _clip((plus_di - minus_di) / 25),
        "Minus_DI": _clip((minus_di - plus_di) / 25),

        "Choppiness_Index": (
            _clip((50 - chop) / 30)
            * _direction(c - sma)
        ),

        "Relative_Vigor_Index": _clip(rvi * 2),
        "Schaff_Trend_Cycle": _clip((stc - 50) / 50),

        # Volatility-only indicators are neutral in directional scoring.
        # Their raw values remain available for risk/regime analysis.
        "ATR": _clip(atr * 0),
        "Bollinger_Bands": _clip(
            _safe_div(
                c - bb_mid,
                2 * bb_std
            ).mask(bb_std == 0, 0.0)
        ),
        "Keltner_Channels": _clip(
            _safe_div(c - kc_mid, 2 * atr)
        ),
        "Donchian_Channels": _clip(
            _safe_div(
                c - (donchian_high + donchian_low) / 2,
                (donchian_high - donchian_low) / 2
            )
        ),
        "Standard_Deviation": _clip(bb_std * 0),
        "Historical_Volatility": _clip(hist_vol * 0),
        "Ulcer_Index": _clip(ulcer * 0),
        "Chaikin_Volatility": _clip(chaikin_vol * 0),

        "OBV": _direction(obv.diff(10)),
        "Chaikin_Money_Flow": _clip(cmf * 3),
        "Accumulation_Distribution": _direction(adl.diff(10)),
        "Force_Index": _direction(force),
        "Ease_of_Movement": _direction(eom),

        # Volume magnitude alone is not a directional signal.
        "Volume_ROC": _clip(volume_roc * 0),
        "Relative_Volume": _clip(relative_volume * 0),

        "VWAP": _direction(_safe_div(c - vwap, atr)),
        "Price_Volume_Trend": _direction(pvt.diff(10)),
        "Negative_Volume_Index": _direction(nvi.diff(10)),
        "Positive_Volume_Index": _direction(pvi.diff(10)),
        "Klinger_Oscillator": _direction(klinger),
    }

    # ==========================================================
    # FOUR SPECIAL EARLY-WARNING FACTORS
    # ==========================================================

    resistance = h.shift(1).rolling(20).max()
    support = l.shift(1).rolling(20).min()

    higher_lows = (
        (l > l.shift(5))
        & (l.shift(5) > l.shift(10))
    )

    lower_highs = (
        (h < h.shift(5))
        & (h.shift(5) < h.shift(10))
    )

    price_flat = (
        c.pct_change(10, fill_method=None).abs() < 0.03
    )

    obv_change = obv.diff(10)

    hidden_accum = pd.Series(
        np.select(
            [
                price_flat & (obv_change > 0),
                price_flat & (obv_change < 0)
            ],
            [1.0, -1.0],
            default=0.0
        ),
        index=df.index
    )

    near_resistance = _safe_div(
        resistance - c,
        c
    ).between(-0.005, 0.02)

    near_support = _safe_div(
        c - support,
        c
    ).between(-0.005, 0.02)

    rising_lows = pd.Series(
        np.select(
            [
                higher_lows & near_resistance,
                lower_highs & near_support
            ],
            [1.0, -1.0],
            default=0.0
        ),
        index=df.index
    )

    candle_range = (h - l).replace(0, np.nan)

    upper_wick = (
        h - pd.concat([o, c], axis=1).max(axis=1)
    )

    lower_wick = (
        pd.concat([o, c], axis=1).min(axis=1) - l
    )

    high_volume = (
        v > v.rolling(20).mean() * 1.3
    )

    absorption = pd.Series(
        np.select(
            [
                (
                    high_volume
                    & near_support
                    & (lower_wick / candle_range > 0.35)
                ),
                (
                    high_volume
                    & near_resistance
                    & (upper_wick / candle_range > 0.35)
                )
            ],
            [1.0, -1.0],
            default=0.0
        ),
        index=df.index
    )

    volume_rising = (
        v.rolling(5).mean()
        > v.rolling(20).mean() * 1.15
    )

    breakout_prep = pd.Series(
        np.select(
            [
                (
                    volume_rising
                    & near_resistance
                    & higher_lows
                    & (c <= resistance * 1.005)
                ),
                (
                    volume_rising
                    & near_support
                    & lower_highs
                    & (c >= support * 0.995)
                )
            ],
            [1.0, -1.0],
            default=0.0
        ),
        index=df.index
    )

    special = {
        "Hidden_Accumulation": hidden_accum,
        "Rising_Lows_Below_Resistance": rising_lows,
        "Resistance_Absorption": absorption,
        "Volume_Breakout_Preparation": breakout_prep,
    }

    signals.update({
        name: _clip(series)
        for name, series in special.items()
    })

    # ==========================================================
    # LATEST VALUES AND SCORES
    # ==========================================================

    if len(values) != 50:
        raise RuntimeError(
            f"Expected 50 indicators, got {len(values)}"
        )

    if len(special) != 4:
        raise RuntimeError(
            f"Expected 4 special factors, got {len(special)}"
        )

    if len(signals) != 54:
        raise RuntimeError(
            f"Expected 54 signals, got {len(signals)}"
        )

    latest_values = {
        name: (
            float(series.iloc[-1])
            if pd.notna(series.iloc[-1])
            else None
        )
        for name, series in values.items()
    }

    latest_special = {
        name: (
            float(series.iloc[-1])
            if pd.notna(series.iloc[-1])
            else None
        )
        for name, series in special.items()
    }

    latest_signals = {
        name: (
            float(series.iloc[-1])
            if pd.notna(series.iloc[-1])
            else None
        )
        for name, series in signals.items()
    }

    long_total = 0.0
    short_total = 0.0
    total_weight = 0.0

    for name, value in latest_signals.items():

        if value is None or not np.isfinite(value):
            continue

        weight = (
            SPECIAL_WEIGHT
            if name in special
            else TECHNICAL_WEIGHT
        )

        long_total += max(value, 0.0) * weight
        short_total += max(-value, 0.0) * weight

        total_weight += weight

    if total_weight <= 0:
        raise ValueError(
            "No valid indicator signals available for scoring"
        )

    long_score = round(
        100 * long_total / total_weight,
        2
    )

    short_score = round(
        100 * short_total / total_weight,
        2
    )

    return {
        "indicator_count": len(values),
        "special_factor_count": len(special),
        "total_factor_count": len(latest_signals),

        "indicator_values": latest_values,
        "special_values": latest_special,
        "signals": latest_signals,

        "long_score": long_score,
        "short_score": short_score,

        "score_weight_units": total_weight,

        "long_label": score_label(long_score),
        "short_label": score_label(short_score),

        "valid_signal_count": sum(
            value is not None
            for value in latest_signals.values()
        ),
    }


def score_label(score):

    if score >= 90:
        return "OLAĞANÜSTÜ SİNYAL"

    if score >= 80:
        return "ÇOK GÜÇLÜ HAZIRLIK"

    if score >= 65:
        return "GÜÇLÜ HAZIRLIK"

    if score >= 50:
        return "TAKİP"

    if score >= 30:
        return "DÜŞÜK"

    return "ZAYIF"


def analyze_timeframes(timeframe_data):
    """
    Analyze all five required timeframes.

    Missing or invalid timeframe data stops final scoring.
    No weight renormalization is permitted.
    """

    if not isinstance(timeframe_data, dict):
        raise TypeError(
            "timeframe_data must be a dictionary"
        )

    required = list(TIMEFRAME_WEIGHTS.keys())

    missing = [
        timeframe
        for timeframe in required
        if timeframe not in timeframe_data
    ]

    if missing:
        raise ValueError(
            f"Missing required timeframes: {missing}"
        )

    results = {}

    long_score = 0.0
    short_score = 0.0

    for timeframe, weight in TIMEFRAME_WEIGHTS.items():

        result = calculate_indicators(
            timeframe_data[timeframe]
        )

        results[timeframe] = result

        long_score += result["long_score"] * weight
        short_score += result["short_score"] * weight

    final_long = round(long_score, 2)
    final_short = round(short_score, 2)

    return {
        "timeframes": results,

        "long_score": final_long,
        "short_score": final_short,

        "long_label": score_label(final_long),
        "short_label": score_label(final_short),

        "available_timeframe_weight": round(
            sum(TIMEFRAME_WEIGHTS.values()),
            4
        ),

        "complete_timeframes": True,
    }
