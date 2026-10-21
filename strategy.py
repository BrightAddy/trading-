from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Signal:
    direction: str
    score: int
    reasons: tuple[str, ...]
    price: float

# ─────────────────────────────────────────────
# Core Indicator Functions
# ─────────────────────────────────────────────

def ema(values, period):
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return a
    out = np.empty_like(a)
    out[0] = a[0]
    k = 2.0 / (period + 1)
    for i in range(1, len(a)):
        out[i] = a[i] * k + out[i - 1] * (1 - k)
    return out

def rsi(values, period=14):
    """Return the full RSI array for the entire series (needed for divergence detection)."""
    a = np.asarray(values, dtype=float)
    if len(a) <= period:
        return np.full(len(a), 50.0)
    d = np.diff(a)
    gains = np.maximum(d, 0.0)
    losses = np.maximum(-d, 0.0)
    rsi_arr = np.full(len(a), 50.0)
    ag = np.mean(gains[:period])
    al = np.mean(losses[:period])
    rsi_arr[period] = 100.0 if al == 0 else 100.0 - (100.0 / (1.0 + ag / al))
    for i in range(period, len(d)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
        rsi_arr[i + 1] = 100.0 if al == 0 else 100.0 - (100.0 / (1.0 + ag / al))
    return rsi_arr

def rsi_single(values, period=14):
    """Return a single RSI value for the end of the series (backward compat)."""
    return float(rsi(values, period)[-1])

def atr(high, low, close, period=14):
    h, l, c = map(lambda x: np.asarray(x, dtype=float), (high, low, close))
    if len(h) <= period:
        return float(np.mean(h - l)) if len(h) > 0 else 0.0
    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    return float(np.mean(tr[-period:]))

def macd(close, fast=12, slow=26, signal_period=9):
    c = np.asarray(close, dtype=float)
    fast_ema = ema(c, fast)
    slow_ema = ema(c, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(close, period=20, num_std=2.0):
    c = np.asarray(close, dtype=float)
    if len(c) < period:
        return c, c, c
    sma = np.convolve(c, np.ones(period)/period, mode='valid')
    sma = np.pad(sma, (period - 1, 0), mode='edge')
    std = np.array([np.std(c[max(0, i - period + 1):i + 1]) for i in range(len(c))])
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower

def adx(high, low, close, period=14):
    h, l, c = map(lambda x: np.asarray(x, dtype=float), (high, low, close))
    if len(h) <= period + 1:
        return 0.0, 0.0, 0.0
    up_move = h[1:] - h[:-1]
    down_move = l[:-1] - l[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))

    atr_val = np.mean(tr[-period:])
    if atr_val == 0:
        return 0.0, 0.0, 0.0

    plus_di = 100.0 * (np.mean(plus_dm[-period:]) / atr_val)
    minus_di = 100.0 * (np.mean(minus_dm[-period:]) / atr_val)
    di_sum = plus_di + minus_di
    dx = 100.0 * (abs(plus_di - minus_di) / di_sum) if di_sum != 0 else 0.0
    return dx, plus_di, minus_di

# ─────────────────────────────────────────────
# NEW Indicators
# ─────────────────────────────────────────────

def stochastic(high, low, close, k_period=14, d_period=3):
    """Stochastic Oscillator — %K and %D values for the last bar."""
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    if len(h) < k_period:
        return 50.0, 50.0
    k_values = []
    for i in range(k_period - 1, len(c)):
        highest = np.max(h[i - k_period + 1:i + 1])
        lowest = np.min(l[i - k_period + 1:i + 1])
        if highest == lowest:
            k_values.append(50.0)
        else:
            k_values.append(100.0 * (c[i] - lowest) / (highest - lowest))
    k_arr = np.array(k_values)
    if len(k_arr) < d_period:
        return float(k_arr[-1]), float(k_arr[-1])
    d_arr = np.convolve(k_arr, np.ones(d_period) / d_period, mode='valid')
    return float(k_arr[-1]), float(d_arr[-1])


def detect_rsi_divergence(close, rsi_values, lookback=20):
    """
    Detect RSI divergence over the recent window.
    Bullish divergence: price makes lower low, RSI makes higher low.
    Bearish divergence: price makes higher high, RSI makes lower high.
    """
    c = np.asarray(close, dtype=float)
    r = np.asarray(rsi_values, dtype=float)
    if len(c) < lookback + 5 or len(r) < lookback + 5:
        return 'none'

    recent_c = c[-lookback:]
    recent_r = r[-lookback:]
    half = lookback // 2

    # Bullish divergence check
    price_low_1 = np.min(recent_c[:half])
    price_low_2 = np.min(recent_c[half:])
    rsi_low_1 = np.min(recent_r[:half])
    rsi_low_2 = np.min(recent_r[half:])
    if price_low_2 < price_low_1 and rsi_low_2 > rsi_low_1:
        return 'bullish_div'

    # Bearish divergence check
    price_high_1 = np.max(recent_c[:half])
    price_high_2 = np.max(recent_c[half:])
    rsi_high_1 = np.max(recent_r[:half])
    rsi_high_2 = np.max(recent_r[half:])
    if price_high_2 > price_high_1 and rsi_high_2 < rsi_high_1:
        return 'bearish_div'

    return 'none'


def detect_candle_patterns(o, h, l, c):
    """Detect reversal/continuation candle patterns on the last 2-3 bars."""
    patterns = []
    if len(o) < 3:
        return patterns

    body = abs(c[-1] - o[-1])
    candle_range = max(h[-1] - l[-1], 1e-12)
    body_ratio = body / candle_range
    upper_wick = h[-1] - max(c[-1], o[-1])
    lower_wick = min(c[-1], o[-1]) - l[-1]

    # Bullish pin bar (hammer)
    if body_ratio < 0.35 and lower_wick > body * 2.0 and upper_wick < body * 0.8:
        patterns.append('bullish_pin_bar')

    # Bearish pin bar (shooting star)
    if body_ratio < 0.35 and upper_wick > body * 2.0 and lower_wick < body * 0.8:
        patterns.append('bearish_pin_bar')

    # Bullish engulfing
    if c[-2] < o[-2] and c[-1] > o[-1] and c[-1] > o[-2] and o[-1] < c[-2]:
        patterns.append('bullish_engulfing')

    # Bearish engulfing
    if c[-2] > o[-2] and c[-1] < o[-1] and c[-1] < o[-2] and o[-1] > c[-2]:
        patterns.append('bearish_engulfing')

    # Strong marubozu (big body, tiny wicks)
    if body_ratio >= 0.75:
        if c[-1] > o[-1]:
            patterns.append('bullish_marubozu')
        else:
            patterns.append('bearish_marubozu')

    # Inside bar breakout
    if h[-2] > h[-1] and l[-2] < l[-1]:
        mid = (h[-2] + l[-2]) / 2
        if c[-1] > o[-1] and c[-1] > mid:
            patterns.append('inside_bar_bull_break')
        elif c[-1] < o[-1] and c[-1] < mid:
            patterns.append('inside_bar_bear_break')

    return patterns


def find_support_resistance(high, low, close, lookback=50):
    """Find swing-high resistance and swing-low support levels."""
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    if len(h) < lookback:
        return [], []

    recent_h = h[-lookback:]
    recent_l = l[-lookback:]
    swing_highs, swing_lows = [], []

    for i in range(2, len(recent_h) - 2):
        if (recent_h[i] > recent_h[i - 1] and recent_h[i] > recent_h[i - 2]
                and recent_h[i] > recent_h[i + 1] and recent_h[i] > recent_h[i + 2]):
            swing_highs.append(float(recent_h[i]))
        if (recent_l[i] < recent_l[i - 1] and recent_l[i] < recent_l[i - 2]
                and recent_l[i] < recent_l[i + 1] and recent_l[i] < recent_l[i + 2]):
            swing_lows.append(float(recent_l[i]))

    price = float(c[-1])
    resistance = sorted([x for x in swing_highs if x > price])[:3]
    support = sorted([x for x in swing_lows if x < price], reverse=True)[:3]
    return support, resistance


# ─────────────────────────────────────────────
# Main Analysis — Multi-Gate Confirmation
# ─────────────────────────────────────────────

def analyse(candles):
    """
    Multi-gate confirmation strategy for high-probability Forex trades.

    ALL 4 mandatory gates must pass + at least one bonus confirmation before
    a signal is emitted.  Anti-exhaustion filters can veto an otherwise valid
    setup.
    """
    if len(candles) < 210:
        return Signal("wait", 0, ("insufficient_history",), 0.0)

    candles = sorted(candles, key=lambda x: x["time"])
    candles = candles[:-1]                          # drop still-forming candle

    o = np.array([float(x["open"]) for x in candles])
    h = np.array([float(x["high"]) for x in candles])
    l = np.array([float(x["low"]) for x in candles])
    c = np.array([float(x["close"]) for x in candles])

    # ── Compute all indicators ───────────────────────────────────────
    e9, e21, e50, e200 = ema(c, 9), ema(c, 21), ema(c, 50), ema(c, 200)

    rsi_7  = rsi(c, 7)
    rsi_14 = rsi(c, 14)
    rsi_21 = rsi(c, 21)
    rsi_7_val  = float(rsi_7[-1])
    rsi_14_val = float(rsi_14[-1])
    rsi_21_val = float(rsi_21[-1])

    volatility   = atr(h, l, c, 14)
    typical_vol  = float(np.mean(h[-60:] - l[-60:]))

    macd_line, macd_sig, macd_hist = macd(c)
    bb_upper, bb_mid, bb_lower     = bollinger_bands(c)
    adx_val, plus_di, minus_di     = adx(h, l, c)
    stoch_k, stoch_d               = stochastic(h, l, c)
    divergence                     = detect_rsi_divergence(c, rsi_14)
    patterns                       = detect_candle_patterns(o, h, l, c)
    support_levels, resistance_levels = find_support_resistance(h, l, c)

    ribbon_spread = abs(e9[-1] - e50[-1]) / max(typical_vol, 1e-12)

    # Recent directional move (exhaustion gauge)
    recent_move  = abs(c[-1] - c[-4]) if len(c) >= 4 else 0
    move_vs_atr  = recent_move / max(volatility, 1e-12)

    body          = abs(c[-1] - o[-1])
    candle_range  = max(h[-1] - l[-1], 1e-12)
    body_ratio    = body / candle_range

    # ── Volatility pre-filter ────────────────────────────────────────
    if typical_vol <= 0 or volatility < typical_vol * 0.4:
        return Signal("wait", 0, ("volatility_too_low",), float(c[-1]))
    if volatility > typical_vol * 2.0:
        return Signal("wait", 0, ("volatility_spike",), float(c[-1]))

    # ── Evaluate each direction through the gate system ──────────────
    for direction in ("call", "put"):
        reasons = []

        # ── GATE 1 — Trend Context (mandatory) ──────────────────────
        gate1 = False
        if direction == "call":
            if e9[-1] > e21[-1] > e50[-1] and c[-1] > e21[-1]:
                gate1 = True
                reasons.append("bullish_trend_confirmed")
            elif c[-2] <= e21[-2] and c[-1] > e21[-1] and e9[-1] > e50[-1]:
                gate1 = True
                reasons.append("ema21_reclaim_trend")
        else:
            if e9[-1] < e21[-1] < e50[-1] and c[-1] < e21[-1]:
                gate1 = True
                reasons.append("bearish_trend_confirmed")
            elif c[-2] >= e21[-2] and c[-1] < e21[-1] and e9[-1] < e50[-1]:
                gate1 = True
                reasons.append("ema21_rejection_trend")
        if not gate1:
            continue

        # ── GATE 2 — Momentum Confirmation (mandatory) ──────────────
        # Stochastic MUST confirm direction for dual/triple.
        # RSI-only fallback allowed only in tight high-confidence zones.
        gate2 = False
        momentum_quality = 0   # 0=not passed, 3=triple, 2=dual, 1=rsi-controlled
        if direction == "call":
            rsi_ok       = 40 <= rsi_14_val <= 72
            rsi_tight    = 48 <= rsi_14_val <= 65  # High-confidence bullish zone
            stoch_ok     = stoch_k > stoch_d and stoch_k < 88
            stoch_neutral = stoch_k > 35  # At least not deeply bearish
            multi_rsi_ok = rsi_7_val >= rsi_14_val - 8
            if rsi_ok and stoch_ok and multi_rsi_ok:
                gate2 = True; momentum_quality = 3; reasons.append("momentum_triple_confirm")
            elif rsi_ok and stoch_ok:
                gate2 = True; momentum_quality = 2; reasons.append("momentum_dual_confirm")
            elif rsi_tight and multi_rsi_ok and stoch_neutral:
                gate2 = True; momentum_quality = 1; reasons.append("momentum_controlled_confirm")
        else:
            rsi_ok       = 28 <= rsi_14_val <= 60
            rsi_tight    = 35 <= rsi_14_val <= 52  # High-confidence bearish zone
            stoch_ok     = stoch_k < stoch_d and stoch_k > 12
            stoch_neutral = stoch_k < 65  # At least not deeply bullish
            multi_rsi_ok = rsi_7_val <= rsi_14_val + 8
            if rsi_ok and stoch_ok and multi_rsi_ok:
                gate2 = True; momentum_quality = 3; reasons.append("momentum_triple_confirm")
            elif rsi_ok and stoch_ok:
                gate2 = True; momentum_quality = 2; reasons.append("momentum_dual_confirm")
            elif rsi_tight and multi_rsi_ok and stoch_neutral:
                gate2 = True; momentum_quality = 1; reasons.append("momentum_controlled_confirm")
        if not gate2:
            continue

        # ── GATE 3 — Price Action (mandatory) ───────────────────────
        # Pin bars removed as standalone trigger (33% WR in backtest)
        gate3 = False
        if direction == "call":
            # Only high-confidence candle patterns (engulfing, marubozu)
            quality_pats = [p for p in patterns if p in (
                'bullish_engulfing', 'bullish_marubozu', 'inside_bar_bull_break')]
            strong       = c[-1] > o[-1] and body_ratio >= 0.50
            breakout     = c[-1] > max(h[-6:-1]) if len(h) >= 6 else False
            close_high   = (c[-1] - l[-1]) / candle_range > 0.60
            consec_bull  = sum(1 for i in range(-3, 0) if c[i] > o[i]) >= 2
            if quality_pats:
                gate3 = True; reasons.extend(quality_pats)
            elif strong and (breakout or close_high):
                gate3 = True; reasons.append("strong_bull_action")
            elif strong and body > typical_vol * 0.4:
                gate3 = True; reasons.append("decisive_bull_candle")
            elif consec_bull and c[-1] > o[-1] and close_high:
                gate3 = True; reasons.append("consecutive_bull_candles")
        else:
            quality_pats = [p for p in patterns if p in (
                'bearish_engulfing', 'bearish_marubozu', 'inside_bar_bear_break')]
            strong       = c[-1] < o[-1] and body_ratio >= 0.50
            breakdown    = c[-1] < min(l[-6:-1]) if len(l) >= 6 else False
            close_low    = (h[-1] - c[-1]) / candle_range > 0.60
            consec_bear  = sum(1 for i in range(-3, 0) if c[i] < o[i]) >= 2
            if quality_pats:
                gate3 = True; reasons.extend(quality_pats)
            elif strong and (breakdown or close_low):
                gate3 = True; reasons.append("strong_bear_action")
            elif strong and body > typical_vol * 0.4:
                gate3 = True; reasons.append("decisive_bear_candle")
            elif consec_bear and c[-1] < o[-1] and close_low:
                gate3 = True; reasons.append("consecutive_bear_candles")
        if not gate3:
            continue

        # ── GATE 4 — Timing / MACD Acceleration (mandatory) ────────
        gate4 = False
        if direction == "call":
            fresh_cross   = macd_hist[-2] <= 0 < macd_hist[-1]
            accelerating  = macd_hist[-1] > 0 and macd_hist[-1] > macd_hist[-2]
            if fresh_cross:
                gate4 = True; reasons.append("macd_fresh_bull_cross")
            elif accelerating:
                gate4 = True; reasons.append("macd_accelerating_bull")
        else:
            fresh_cross   = macd_hist[-2] >= 0 > macd_hist[-1]
            accelerating  = macd_hist[-1] < 0 and macd_hist[-1] < macd_hist[-2]
            if fresh_cross:
                gate4 = True; reasons.append("macd_fresh_bear_cross")
            elif accelerating:
                gate4 = True; reasons.append("macd_accelerating_bear")
        if not gate4:
            continue

        # ── Anti-exhaustion vetoes ──────────────────────────────────
        if direction == "call" and rsi_14_val > 75:
            continue
        if direction == "put"  and rsi_14_val < 25:
            continue
        if move_vs_atr > 2.0:
            continue
        if direction == "call" and len(macd_hist) >= 3:
            if macd_hist[-1] > 0 and macd_hist[-1] < macd_hist[-2] < macd_hist[-3]:
                continue
        if direction == "put" and len(macd_hist) >= 3:
            if macd_hist[-1] < 0 and macd_hist[-1] > macd_hist[-2] > macd_hist[-3]:
                continue
        if ribbon_spread < 0.2:
            continue

        # ── GATE 5 — Micro-regime consistency check ─────────────────
        # Reject trades that fight the micro-regime direction
        # (e.g. CALL during a micro_pullback, or PUT during micro_bounce)
        if direction == "call" and move_vs_atr > 1.2:
            # Price just surged — likely to retrace, not continue
            if len(c) >= 5 and c[-1] > c[-5] and (c[-1] - c[-5]) / max(volatility, 1e-12) > 1.2:
                continue
        if direction == "put" and move_vs_atr > 1.2:
            if len(c) >= 5 and c[-1] < c[-5] and (c[-5] - c[-1]) / max(volatility, 1e-12) > 1.2:
                continue

        # ── Bonus gates → build confidence score ────────────────────
        score  = 70          # base for passing all 4 mandatory gates
        bonus  = 0

        # ADX
        if adx_val >= 25:
            if direction == "call" and plus_di > minus_di:
                bonus += 10; reasons.append("adx_strong_bull")
            elif direction == "put" and minus_di > plus_di:
                bonus += 10; reasons.append("adx_strong_bear")

        # Bollinger Band context
        if direction == "call":
            if c[-1] <= bb_lower[-1] * 1.005:
                bonus += 10; reasons.append("bb_lower_support")
            elif bb_lower[-1] < c[-1] < bb_mid[-1]:
                bonus += 5;  reasons.append("bb_lower_half")
        else:
            if c[-1] >= bb_upper[-1] * 0.995:
                bonus += 10; reasons.append("bb_upper_resistance")
            elif bb_mid[-1] < c[-1] < bb_upper[-1]:
                bonus += 5;  reasons.append("bb_upper_half")

        # RSI divergence
        if direction == "call" and divergence == "bullish_div":
            bonus += 10; reasons.append("rsi_bullish_divergence")
        if direction == "put"  and divergence == "bearish_div":
            bonus += 10; reasons.append("rsi_bearish_divergence")

        # Near support / resistance
        if support_levels and direction == "call":
            dist = (c[-1] - support_levels[0]) / max(volatility, 1e-12)
            if dist < 1.5:
                bonus += 8; reasons.append("near_support")
        if resistance_levels and direction == "put":
            dist = (resistance_levels[0] - c[-1]) / max(volatility, 1e-12)
            if dist < 1.5:
                bonus += 8; reasons.append("near_resistance")

        # Full EMA stack — data shows 42.9% WR when momentum isn't triple-confirmed
        # This means the trend is extended and exhaustion risk is high
        full_stack_bull = (direction == "call" and e9[-1] > e21[-1] > e50[-1] > e200[-1])
        full_stack_bear = (direction == "put"  and e9[-1] < e21[-1] < e50[-1] < e200[-1])
        if full_stack_bull or full_stack_bear:
            if momentum_quality >= 3:
                bonus += 5; reasons.append("full_stack_strong_momentum")
            else:
                # Extended trend + weak momentum = VETO (42.9% WR in backtest)
                continue

        # EMAs trending correctly over last 5 bars
        if direction == "call" and e21[-1] > e21[-6] and e50[-1] > e50[-6]:
            bonus += 5; reasons.append("emas_rising")
        if direction == "put"  and e21[-1] < e21[-6] and e50[-1] < e50[-6]:
            bonus += 5; reasons.append("emas_falling")

        # Require meaningful bonus confirmation
        # If momentum was only rsi-confirmed (weakest), require higher bonus
        min_bonus = 10 if momentum_quality <= 1 else 5
        if bonus < min_bonus:
            continue

        score = min(score + bonus, 100)
        return Signal(direction, score, tuple(reasons), float(c[-1]))

    # No direction passed all gates
    return Signal("wait", 0, ("no_confirmed_setup",), float(c[-1]))

# Import new institutional strategies
from strategies.mtf_pullback import analyse_mtf_pullback
from strategies.session_breakout import analyse_session_breakout

