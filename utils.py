import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def append_trade_log(path: Path, row: dict):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow([
                "timestamp",
                "direction",
                "score",
                "reasons",
                "entry",
                "exit",
                "win",
                "pnl",
            ])
        writer.writerow([
            row["timestamp"],
            row["direction"],
            row["score"],
            "|".join(row["reasons"]),
            row["entry"],
            row["exit"],
            row["win"],
            row["pnl"],
        ])


def summarize_trade_log(path: Path):
    import csv

    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    totals = {
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "total_pnl": 0.0,
        "score_sum": 0.0,
        "directions": defaultdict(int),
        "reasons": defaultdict(int),
    }

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            totals["trades"] += 1
            pnl = float(row["pnl"])
            totals["total_pnl"] += pnl
            totals["wins"] += 1 if pnl > 0 else 0
            totals["losses"] += 1 if pnl <= 0 else 0
            totals["score_sum"] += float(row["score"])
            totals["directions"][row["direction"]] += 1
            for reason in row["reasons"].split("|"):
                totals["reasons"][reason] += 1

    totals["win_rate"] = totals["wins"] / totals["trades"] * 100 if totals["trades"] else 0.0
    totals["average_score"] = totals["score_sum"] / totals["trades"] if totals["trades"] else 0.0
    return totals


def print_trade_summary(summary: dict):
    print("Trade log summary")
    print("-----------------")
    print(f"Trades: {summary['trades']}")
    print(f"Wins: {summary['wins']}  Losses: {summary['losses']}")
    print(f"Win rate: {summary['win_rate']:.2f}%")
    print(f"Net PnL: {summary['total_pnl']:+.2f}")
    print(f"Average score: {summary['average_score']:.1f}")
    print("Direction counts:")
    for direction, count in summary["directions"].items():
        print(f"  {direction}: {count}")
    print("Top reasons:")
    for reason, count in sorted(summary["reasons"].items(), key=lambda x: -x[1])[:12]:
        print(f"  {reason}: {count}")
