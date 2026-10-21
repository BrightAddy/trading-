"""
Multi-Timeframe (MTF) Pullback Strategy (H1 Macro Trend + M5 Value Pullback Entry)
---------------------------------------------------------------------------------
Institutional logic:
1. Higher Timeframe (H1) establishes the macro trend direction (50 & 200 EMA + structure).
2. Lower Timeframe (M5) waits for price to pull back to dynamic support/resistance (21/50 EMA value zone, RSI discount).
3. Enters on price-action confirmation (pin bar, engulfing, or momentum hook) at the discount price.
4. Provides asymmetric 1:2 or 1:3 Risk-to-Reward.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass(frozen=True)
class Signal:
    direction: str       # "call", "put", or "wait"
    score: int          # 0 - 100
    reasons: tuple[str, ...]
    price: float


def ema(values: np.ndarray, period: int) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return a
    out = np.empty_like(a)
    out[0] = a[0]
    k = 2.0 / (period + 1)
    for i in range(1, len(a)):
        out[i] = a[i] * k + out[i - 1] * (1 - k)
    return out


def rsi(values: np.ndarray, period: int = 14) -> np.ndarray:
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


def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
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


def detect_candle_reversal(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> List[str]:
    """Detect reversal signals on the last 2 completed candles."""
    patterns = []
    if len(o) < 3:
        return patterns

    body = abs(c[-1] - o[-1])
    candle_range = max(h[-1] - l[-1], 1e-12)
    body_ratio = body / candle_range
    upper_wick = h[-1] - max(c[-1], o[-1])
    lower_wick = min(c[-1], o[-1]) - l[-1]

    # Hammer / Bullish Pin Bar
    if body_ratio < 0.35 and lower_wick >= body * 1.8 and upper_wick <= body * 0.8:
        patterns.append("bullish_pin_bar")

    # Shooting Star / Bearish Pin Bar
    if body_ratio < 0.35 and upper_wick >= body * 1.8 and lower_wick <= body * 0.8:
        patterns.append("bearish_pin_bar")

    # Bullish Engulfing
    if c[-2] < o[-2] and c[-1] > o[-1] and c[-1] >= o[-2] and o[-1] <= c[-2]:
        patterns.append("bullish_engulfing")

    # Bearish Engulfing
    if c[-2] > o[-2] and c[-1] < o[-1] and c[-1] <= o[-2] and o[-1] >= c[-2]:
        patterns.append("bearish_engulfing")

    # Strong decisive close
    if body_ratio >= 0.50:
        if c[-1] > o[-1] and (c[-1] - l[-1]) / candle_range > 0.70:
            patterns.append("strong_bull_close")
        elif c[-1] < o[-1] and (h[-1] - c[-1]) / candle_range > 0.70:
            patterns.append("strong_bear_close")

    return patterns


def get_h1_trend_bias(h1_candles: list) -> str:
    """Determine H1 Higher Timeframe trend bias."""
    if not h1_candles or len(h1_candles) < 30:
        return "neutral"

    closes = np.array([float(x["close"]) for x in h1_candles])
    e21 = ema(closes, 21)
    e50 = ema(closes, 50)
    e200 = ema(closes, min(200, len(closes) - 1)) if len(closes) >= 60 else ema(closes, 50)

    c_last = closes[-1]
    e50_last = e50[-1]
    e200_last = e200[-1]
    e21_last = e21[-1]

    # Check slope of EMA50 over last 5 bars
    slope_e50 = (e50[-1] - e50[-5]) if len(e50) >= 5 else 0

    if (e50_last > e200_last or c_last > e50_last) and e21_last >= e50_last and slope_e50 >= 0:
        return "bull"
    elif (e50_last < e200_last or c_last < e50_last) and e21_last <= e50_last and slope_e50 <= 0:
        return "bear"
    return "neutral"


def analyse_mtf_pullback(h1_candles: list, m5_candles: list) -> Signal:
    """
    Main Multi-Timeframe Pullback Strategy Evaluator.
    Returns Signal("call" | "put" | "wait", score, reasons, price).
    """
    if not m5_candles or len(m5_candles) < 60:
        return Signal("wait", 0, ("insufficient_m5_history",), 0.0)

    # Use only closed candles (drop the currently forming bar)
    m5 = sorted(m5_candles, key=lambda x: x["time"])[:-1]
    h1 = sorted(h1_candles, key=lambda x: x["time"])[:-1] if h1_candles else []

    o = np.array([float(x["open"]) for x in m5])
    h = np.array([float(x["high"]) for x in m5])
    l = np.array([float(x["low"]) for x in m5])
    c = np.array([float(x["close"]) for x in m5])

    last_price = float(c[-1])

    # 1. Higher Timeframe Bias
    h1_bias = get_h1_trend_bias(h1)
    if h1_bias == "neutral":
        # Fallback: if H1 candles not available, evaluate M5 macro trend (EMA 200)
        e200_m5 = ema(c, min(200, len(c)-1))
        e50_m5 = ema(c, 50)
        if e50_m5[-1] > e200_m5[-1] and c[-1] > e200_m5[-1]:
            h1_bias = "bull"
        elif e50_m5[-1] < e200_m5[-1] and c[-1] < e200_m5[-1]:
            h1_bias = "bear"
        else:
            return Signal("wait", 0, ("macro_trend_neutral",), last_price)

    # 2. Lower Timeframe (M5) Indicators
    e9 = ema(c, 9)
    e21 = ema(c, 21)
    e50 = ema(c, 50)
    rsi_vals = rsi(c, 14)
    rsi_now = float(rsi_vals[-1])
    rsi_prev = float(rsi_vals[-2])
    stoch_k, stoch_d = stochastic(h, l, c)
    patterns = detect_candle_reversal(o, h, l, c)

    reasons = [f"h1_trend_{h1_bias}"]
    score = 70

    # ─────────────────────────────────────────────────────────────
    # BUY SETUP (Pullback in Bullish Macro Trend)
    # ─────────────────────────────────────────────────────────────
    if h1_bias == "bull":
        # Condition A: Price has pulled back to or near the EMA 21 / EMA 50 value zone
        pullback_tested = (min(l[-3:]) <= e21[-1] * 1.001) or (min(l[-3:]) <= e50[-1] * 1.002)
        if not pullback_tested:
            return Signal("wait", 0, ("waiting_for_bull_pullback",), last_price)

        # Condition B: RSI pulled back into value zone (RSI was <= 45 recently or is currently <= 50)
        rsi_discount = (min(rsi_vals[-4:]) <= 48.0)
        if not rsi_discount:
            return Signal("wait", 0, ("rsi_not_in_discount",), last_price)

        # Condition C: Reversal Confirmation on closed candle
        # Either a recognized candle pattern, or RSI turning up + green candle
        reversal_candle = any(p in patterns for p in ("bullish_pin_bar", "bullish_engulfing", "strong_bull_close"))
        rsi_hook_up = (rsi_now > rsi_prev and rsi_now >= 38.0 and c[-1] > o[-1])
        stoch_turning_up = (stoch_k > stoch_d and stoch_k < 75.0)

        if not (reversal_candle or (rsi_hook_up and stoch_turning_up)):
            return Signal("wait", 0, ("waiting_for_reversal_confirmation",), last_price)

        # Score calculations
        if reversal_candle:
            reasons.extend(patterns)
            score += 10
        if rsi_hook_up:
            reasons.append("rsi_momentum_hook_up")
            score += 10
        if stoch_turning_up:
            reasons.append("stoch_oversold_bull_cross")
            score += 10
        if c[-1] > e21[-1]:
            reasons.append("ema21_reclaimed")
            score += 5

        return Signal("call", min(score, 100), tuple(reasons), last_price)

    # ─────────────────────────────────────────────────────────────
    # SELL SETUP (Pullback/Rally in Bearish Macro Trend)
    # ─────────────────────────────────────────────────────────────
    elif h1_bias == "bear":
        # Condition A: Price rallied back to or near the EMA 21 / EMA 50 resistance zone
        pullback_tested = (max(h[-3:]) >= e21[-1] * 0.999) or (max(h[-3:]) >= e50[-1] * 0.998)
        if not pullback_tested:
            return Signal("wait", 0, ("waiting_for_bear_pullback",), last_price)

        # Condition B: RSI rallied into premium zone (RSI was >= 52 recently or is currently >= 50)
        rsi_premium = (max(rsi_vals[-4:]) >= 52.0)
        if not rsi_premium:
            return Signal("wait", 0, ("rsi_not_in_premium",), last_price)

        # Condition C: Reversal Confirmation on closed candle
        reversal_candle = any(p in patterns for p in ("bearish_pin_bar", "bearish_engulfing", "strong_bear_close"))
        rsi_hook_down = (rsi_now < rsi_prev and rsi_now <= 62.0 and c[-1] < o[-1])
        stoch_turning_down = (stoch_k < stoch_d and stoch_k > 25.0)

        if not (reversal_candle or (rsi_hook_down and stoch_turning_down)):
            return Signal("wait", 0, ("waiting_for_reversal_confirmation",), last_price)

        if reversal_candle:
            reasons.extend(patterns)
            score += 10
        if rsi_hook_down:
            reasons.append("rsi_momentum_hook_down")
            score += 10
        if stoch_turning_down:
            reasons.append("stoch_overbought_bear_cross")
            score += 10
        if c[-1] < e21[-1]:
            reasons.append("ema21_rejected")
            score += 5

        return Signal("put", min(score, 100), tuple(reasons), last_price)

    return Signal("wait", 0, ("no_setup",), last_price)
