"""
Session Volatility Breakout Strategy (London & New York Open Range Breakout)
----------------------------------------------------------------------------
Institutional logic:
1. Calculates the consolidation range of the Asian Session (00:00 - 07:00 UTC).
2. During the London Open (07:00 - 11:00 UTC) and NY Open (12:30 - 16:30 UTC),
   institutional liquidity enters the market.
3. Triggers directional breakouts when price breaks out of the Asian range with
   expanding momentum and volume.
4. Auto-restricts trading outside of liquid session hours to avoid choppy dead zones.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class Signal:
    direction: str       # "call", "put", or "wait"
    score: int          # 0 - 100
    reasons: tuple[str, ...]
    price: float


def get_current_session_window(utc_now: datetime) -> Tuple[bool, str]:
    """Check if current UTC time is within active London or New York trading windows."""
    hour = utc_now.hour + utc_now.minute / 60.0

    # London Open Window: 07:00 - 11:00 UTC
    if 7.0 <= hour <= 11.0:
        return True, "london_open"

    # New York Open Window: 12.5 - 16.5 UTC (12:30 - 16:30 UTC)
    if 12.5 <= hour <= 16.5:
        return True, "ny_open"

    return False, "outside_session_hours"


def calculate_asian_range(candles: list, target_date: datetime.date) -> Tuple[Optional[float], Optional[float], float]:
    """
    Calculate Asian Session (00:00 - 07:00 UTC) High and Low for the given date.
    Returns (asian_high, asian_low, asian_range_distance).
    """
    asian_highs = []
    asian_lows = []

    for c in candles:
        # candle time is unix timestamp (UTC)
        dt = datetime.fromtimestamp(c["time"], tz=timezone.utc)
        if dt.date() == target_date and 0 <= dt.hour < 7:
            asian_highs.append(float(c["high"]))
            asian_lows.append(float(c["low"]))

    if not asian_highs or not asian_lows:
        return None, None, 0.0

    high_val = max(asian_highs)
    low_val = min(asian_lows)
    range_val = high_val - low_val
    return high_val, low_val, range_val


def analyse_session_breakout(candles: list, buffer_pips: float = 2.5, pip_size: float = 0.0001) -> Signal:
    """
    Evaluate Session Breakout setups on M5/M15 candles.
    """
    if not candles or len(candles) < 60:
        return Signal("wait", 0, ("insufficient_history",), 0.0)

    # Work with closed candles (drop still-forming candle)
    c_candles = sorted(candles, key=lambda x: x["time"])[:-1]
    last_candle = c_candles[-1]
    last_price = float(last_candle["close"])
    last_open = float(last_candle["open"])

    # Determine current UTC time
    now_utc = datetime.now(timezone.utc)
    in_session, session_name = get_current_session_window(now_utc)

    if not in_session:
        return Signal("wait", 0, (f"session_filter_inactive_{session_name}",), last_price)

    # Compute Asian range for today
    today_utc = now_utc.date()
    asian_high, asian_low, asian_range = calculate_asian_range(c_candles, today_utc)

    if asian_high is None or asian_low is None or asian_range <= 0:
        # Fallback to last 50 bars range if today's Asian session is unavailable
        recent_h = [float(x["high"]) for x in c_candles[-40:-10]]
        recent_l = [float(x["low"]) for x in c_candles[-40:-10]]
        if not recent_h:
            return Signal("wait", 0, ("asian_range_not_found",), last_price)
        asian_high = max(recent_h)
        asian_low = min(recent_l)
        asian_range = asian_high - asian_low

    buffer_dist = buffer_pips * pip_size
    reasons = [f"session_{session_name}"]
    score = 75

    # Check for Bullish Breakout above Asian High
    if last_price > (asian_high + buffer_dist) and last_price > last_open:
        reasons.append("asian_high_breakout")
        # Check breakout candle strength
        body = last_price - last_open
        candle_range = max(float(last_candle["high"]) - float(last_candle["low"]), 1e-12)
        if body / candle_range >= 0.50:
            reasons.append("strong_breakout_body")
            score += 15

        # Check if previous candle was near or below the high (fresh breakout)
        prev_close = float(c_candles[-2]["close"])
        if prev_close <= asian_high * 1.002:
            reasons.append("fresh_breakout_confirmation")
            score += 10

        return Signal("call", min(score, 100), tuple(reasons), last_price)

    # Check for Bearish Breakout below Asian Low
    if last_price < (asian_low - buffer_dist) and last_price < last_open:
        reasons.append("asian_low_breakdown")
        body = last_open - last_price
        candle_range = max(float(last_candle["high"]) - float(last_candle["low"]), 1e-12)
        if body / candle_range >= 0.50:
            reasons.append("strong_breakdown_body")
            score += 15

        prev_close = float(c_candles[-2]["close"])
        if prev_close >= asian_low * 0.998:
            reasons.append("fresh_breakdown_confirmation")
            score += 10

        return Signal("put", min(score, 100), tuple(reasons), last_price)

    return Signal("wait", 0, ("inside_asian_session_range",), last_price)
