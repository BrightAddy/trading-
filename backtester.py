import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean

from config import SETTINGS
from regime import regime_filter, signal_matches_regime
from strategy import analyse, atr


def parse_timestamp(value: str | None):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return int(float(value))
        except ValueError:
            pass
    try:
        return int(datetime.fromisoformat(value).timestamp())
    except ValueError as exc:
        raise ValueError(f"Unable to parse timestamp '{value}'") from exc


def load_historical_candles(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Historical file has no header row.")
        header = [name.strip().lower() for name in reader.fieldnames]
        required = {"open", "high", "low", "close"}
        if not required.issubset(header):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
        for raw in reader:
            row = {key.strip().lower(): val for key, val in raw.items() if key}
            ts = row.get("timestamp") or row.get("time") or row.get("date") or row.get("datetime")
            rows.append({
                "time": parse_timestamp(ts) if ts else len(rows),
                "open": float(row["open"]),
                "high": float(row.get("high", row.get("max"))),
                "low": float(row.get("low", row.get("min"))),
                "close": float(row["close"]),
            })
    return rows


def simulate_backtest(candles, minimum_score, use_regime_filter,
                      sl_atr_mult=None, tp_atr_mult=None,
                      candle_horizon=None, lot_size=None):
    """
    Forex-style backtest: for each signal, compute ATR-based SL and TP,
    then walk forward through subsequent candles to see which level is
    hit first.  If neither is hit within `candle_horizon` bars, the trade
    is closed at the last bar's close price.
    """
    sl_atr_mult = sl_atr_mult or SETTINGS.sl_atr_multiplier
    tp_atr_mult = tp_atr_mult or SETTINGS.tp_atr_multiplier
    candle_horizon = candle_horizon or SETTINGS.backtest_candle_horizon
    lot_size = lot_size or SETTINGS.backtest_lot_size

    records = []
    reasons_counter = Counter()
    direction_counts = Counter()
    regimes_counter = Counter()
    scores = []

    for index in range(SETTINGS.candle_count - 1, len(candles) - candle_horizon):
        window = candles[: index + 1]
        if len(window) < SETTINGS.candle_count:
            continue

        signal = analyse(window)
        if signal.direction == "wait" or signal.score < minimum_score:
            continue

        regime = regime_filter(window)
        if use_regime_filter and not signal_matches_regime(signal.direction, regime):
            continue

        # ── Compute ATR-based SL & TP (same logic as bot.py) ─────
        highs  = [c["high"] for c in window[-30:]]
        lows   = [c["low"]  for c in window[-30:]]
        closes = [c["close"] for c in window[-30:]]
        atr_val = atr(highs, lows, closes, 14)
        if atr_val <= 0:
            atr_val = (max(highs) - min(lows)) / 14

        entry_price = window[-1]["close"]
        sl_distance = atr_val * sl_atr_mult
        tp_distance = atr_val * tp_atr_mult

        if signal.direction == "call":
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:  # put
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance

        # ── Walk forward to determine outcome ────────────────────
        exit_price = None
        win = None
        for future_idx in range(index + 1, min(index + 1 + candle_horizon, len(candles))):
            bar = candles[future_idx]
            if signal.direction == "call":
                # Check SL hit first (worst case: wick touches SL before TP)
                if bar["low"] <= sl_price:
                    exit_price = sl_price
                    win = False
                    break
                if bar["high"] >= tp_price:
                    exit_price = tp_price
                    win = True
                    break
            else:  # put
                if bar["high"] >= sl_price:
                    exit_price = sl_price
                    win = False
                    break
                if bar["low"] <= tp_price:
                    exit_price = tp_price
                    win = True
                    break

        # If neither SL nor TP hit within horizon, close at last available bar
        if exit_price is None:
            last_bar_idx = min(index + candle_horizon, len(candles) - 1)
            exit_price = candles[last_bar_idx]["close"]
            if signal.direction == "call":
                win = exit_price > entry_price
            else:
                win = exit_price < entry_price

        # ── Calculate P&L in price units × lot size ──────────────
        if signal.direction == "call":
            pnl = (exit_price - entry_price) * lot_size * 100000  # standard lot = 100k units
        else:
            pnl = (entry_price - exit_price) * lot_size * 100000

        records.append({
            "timestamp": window[-1]["time"],
            "direction": signal.direction,
            "score": signal.score,
            "reasons": signal.reasons,
            "regime": regime,
            "entry": entry_price,
            "exit": exit_price,
            "sl": sl_price,
            "tp": tp_price,
            "win": win,
            "pnl": pnl,
        })
        scores.append(signal.score)
        regimes_counter[regime] += 1
        for reason in signal.reasons:
            reasons_counter[reason] += 1
        direction_counts[signal.direction] += 1

    total = len(records)
    wins = sum(1 for row in records if row["win"])
    losses = total - wins
    total_pnl = sum(row["pnl"] for row in records)
    average_score = mean(scores) if scores else 0
    win_rate = wins / total * 100 if total else 0
    average_win = mean([row["pnl"] for row in records if row["win"]]) if wins else 0
    average_loss = mean([row["pnl"] for row in records if not row["win"]]) if losses else 0
    gross_wins = sum(row["pnl"] for row in records if row["win"])
    gross_losses = abs(sum(row["pnl"] for row in records if not row["win"]))
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else float("inf")
    expectancy = ((wins / total) * average_win + (losses / total) * average_loss) if total else 0

    # Advanced risk metrics
    pnls = [r["pnl"] for r in records]
    cum_pnl = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for p in pnls:
        cum_pnl += p
        if cum_pnl > peak:
            peak = cum_pnl
        dd = peak - cum_pnl
        if dd > max_drawdown:
            max_drawdown = dd

    std_dev = (sum((p - (total_pnl / total)) ** 2 for p in pnls) / total) ** 0.5 if total > 1 else 0.0
    sharpe_ratio = ((total_pnl / total) / std_dev * (total ** 0.5)) if std_dev > 0 else 0.0

    return {
        "records": records,
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "average_score": average_score,
        "average_win": average_win,
        "average_loss": average_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "reasons": reasons_counter,
        "direction_counts": direction_counts,
        "regimes": regimes_counter,
    }


def format_currency(value: float):
    return f"{value:+,.2f}"


def main():
    parser = argparse.ArgumentParser(description="Backtest the Forex strategy on historical candle data (ATR-based SL/TP).")
    parser.add_argument("file", help="Path to historical candle CSV")
    parser.add_argument("--min-score", type=int, default=SETTINGS.minimum_score, help="Minimum score to place a trade")
    parser.add_argument("--sl-mult", type=float, default=SETTINGS.sl_atr_multiplier, help="Stop Loss ATR multiplier")
    parser.add_argument("--tp-mult", type=float, default=SETTINGS.tp_atr_multiplier, help="Take Profit ATR multiplier")
    parser.add_argument("--horizon", type=int, default=SETTINGS.backtest_candle_horizon, help="Max candles to wait for SL/TP")
    parser.add_argument("--lot-size", type=float, default=SETTINGS.backtest_lot_size, help="Lot size for P&L calculation")
    parser.add_argument("--regime-filter", action="store_true", help="Only take trades matching regime")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(f"Historical file not found: {path}")

    candles = load_historical_candles(path)
    results = simulate_backtest(
        candles, args.min_score, args.regime_filter,
        sl_atr_mult=args.sl_mult, tp_atr_mult=args.tp_mult,
        candle_horizon=args.horizon, lot_size=args.lot_size
    )

    print("=======================================")
    print("  MT5 Forex Backtest Summary")
    print("=======================================")
    print(f"  Candles loaded:    {len(candles)}")
    print(f"  Trades simulated:  {results['total_trades']}")
    print(f"  Wins: {results['wins']}  |  Losses: {results['losses']}")
    print(f"  Win rate:          {results['win_rate']:.2f}%")
    print(f"  Net P&L:           {format_currency(results['total_pnl'])}")
    print(f"  Avg win:           {format_currency(results['average_win'])}")
    print(f"  Avg loss:          {format_currency(results['average_loss'])}")
    print(f"  Profit factor:     {results['profit_factor']:.2f}")
    print(f"  Expectancy:        {format_currency(results['expectancy'])}")
    print(f"  Max drawdown:      {format_currency(results['max_drawdown'])}")
    print(f"  Sharpe ratio:      {results['sharpe_ratio']:.2f}")
    print(f"  Avg score:         {results['average_score']:.1f}")
    print(f"  SL mult: {args.sl_mult}x ATR  |  TP mult: {args.tp_mult}x ATR")
    print(f"  Lot size: {args.lot_size}  |  Horizon: {args.horizon} candles")
    print("---------------------------------------")
    print("Direction counts:")
    for direction, count in results["direction_counts"].items():
        print(f"  {direction}: {count}")
    print("Regime counts:")
    for regime, count in results["regimes"].most_common(12):
        print(f"  {regime}: {count}")
    print("\nTop signal reasons:")
    for reason, count in results["reasons"].most_common(12):
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
