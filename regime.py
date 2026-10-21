import numpy as np


def slope(values, period):
    if len(values) < period + 1:
        return 0.0
    y = np.arange(period, dtype=float)
    x = np.asarray(values[-period:], dtype=float)
    coeffs = np.polyfit(y, x, 1)
    return float(coeffs[0])


def regime_filter(candles):
    """
    Classify the market regime using macro (200-bar) and micro (20-bar)
    trend/volatility analysis.

    Returns a string like 'bull_normal', 'bear_high', 'range', 'flat_low',
    or compound micro-regimes like 'bull_normal__micro_pullback'.
    """
    if len(candles) < 100:
        return "unknown"

    closes = np.array([float(x["close"]) for x in candles])
    highs  = np.array([float(x["high"]) for x in candles])
    lows   = np.array([float(x["low"]) for x in candles])

    # ── Macro regime (longer window) ─────────────────────────────
    ma50  = np.mean(closes[-50:])
    ma200 = np.mean(closes[-200:]) if len(closes) >= 200 else ma50

    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )
    atr_val   = np.mean(tr[-14:])
    avg_range = np.mean(highs[-20:] - lows[-20:])

    macro_trend = "bull" if ma50 > ma200 else "bear" if ma50 < ma200 else "flat"
    momentum    = slope(closes, 20)
    volatility  = (
        "high" if atr_val > avg_range * 1.3
        else "low" if atr_val < avg_range * 0.7
        else "normal"
    )

    if abs(momentum) < avg_range * 0.001:
        macro_trend = "flat"

    # Range detection
    if volatility == "low" and macro_trend == "flat":
        macro_regime = "range"
    else:
        macro_regime = f"{macro_trend}_{volatility}"

    # ── Micro regime (last ~20 bars, detects pullbacks / pauses) ──
    if len(closes) >= 25:
        micro_slope = slope(closes, 10)
        micro_ma    = np.mean(closes[-10:])
        macro_ma    = np.mean(closes[-30:])

        if macro_trend == "bull" and micro_slope < 0:
            macro_regime += "__micro_pullback"
        elif macro_trend == "bear" and micro_slope > 0:
            macro_regime += "__micro_bounce"
        elif macro_trend in ("flat", "range") and abs(micro_slope) > avg_range * 0.005:
            if micro_slope > 0:
                macro_regime += "__micro_uptick"
            else:
                macro_regime += "__micro_downtick"

    return macro_regime


def signal_matches_regime(direction: str, regime: str) -> bool:
    """
    Decide whether a signal direction is compatible with the detected regime.

    Key change: range-bound markets are no longer blanket-rejected. Instead,
    mean-reversion setups (which the new strategy already validates through
    Bollinger Bands + support/resistance gates) are allowed.
    """
    if regime == "unknown":
        return False

    base = regime.split("__")[0]        # strip micro component
    micro = regime.split("__")[1] if "__" in regime else ""

    # Strong trending regimes — only allow trend-following direction
    if base.startswith("bull") and direction == "call":
        # Reject if micro shows a pullback that hasn't reversed yet
        if micro == "micro_pullback":
            return False          # Wait for the pullback to finish
        return True
    if base.startswith("bear") and direction == "put":
        if micro == "micro_bounce":
            return False
        return True

    # Range / flat regimes — allow trades only if micro-regime confirms
    if base in ("range", "flat_low", "flat_normal", "flat_high"):
        # In range markets, allow mean-reversion trades when micro shows direction
        if direction == "call" and micro == "micro_uptick":
            return True
        if direction == "put" and micro == "micro_downtick":
            return True
        # If no micro-regime, still allow — the strategy gates will be selective
        if not micro:
            return True
        return False

    # Contra-trend signals (e.g. PUT in a bull market) — reject
    return False
