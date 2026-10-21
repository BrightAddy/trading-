import argparse
import base64
import csv
import html
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from backtester import load_historical_candles, simulate_backtest
from config import SETTINGS


def build_score_pnl_figure(results):
    scores = [row["score"] for row in results["records"]]
    pnl = [row["pnl"] for row in results["records"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(scores, pnl, alpha=0.5)
    ax.set_title("Backtest trade P&L vs signal score")
    ax.set_xlabel("Signal score")
    ax.set_ylabel("Trade P&L ($)")
    ax.grid(True)
    return fig


def plot_pnl_vs_score(results, output_path: Path):
    fig = build_score_pnl_figure(results)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_equity_curve_figure(results):
    pnl = np.array([row["pnl"] for row in results["records"]], dtype=float)
    if pnl.size == 0:
        raise ValueError("No trades available for equity curve plotting.")
    cumulative = np.cumsum(pnl)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cumulative, marker="o", markersize=3, linestyle="-", color="tab:blue")
    ax.set_title("Portfolio equity curve")
    ax.set_xlabel("Trade index")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.grid(True)
    return fig


def plot_equity_curve(results, output_path: Path):
    fig = build_equity_curve_figure(results)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_score_heatmap_figure(results, bin_size: int = 5):
    scores = np.array([row["score"] for row in results["records"]], dtype=float)
    win = np.array([row["win"] for row in results["records"]], dtype=int)
    if scores.size == 0:
        raise ValueError("No trades available for heatmap plotting.")
    bins = np.arange(0, 101, bin_size)
    bin_indices = np.digitize(scores, bins, right=True)
    heatmap = np.zeros((2, len(bins) + 1), dtype=int)
    for idx, is_win in zip(bin_indices, win):
        row = 1 if is_win else 0
        heatmap[row, idx] += 1

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(heatmap, cmap="magma", aspect="auto")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Loss", "Win"])
    xticks = np.linspace(0, len(bins), min(len(bins), 11), endpoint=False, dtype=int)
    xtick_labels = [f"{bins[i]}-{bins[i]+bin_size}" for i in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=45)
    ax.set_xlabel("Signal score bucket")
    ax.set_ylabel("Outcome")
    ax.set_title("Score vs outcome heatmap")
    fig.colorbar(im, ax=ax, label="Trade count")
    fig.tight_layout()
    return fig


def plot_score_heatmap(results, output_path: Path, bin_size: int = 5):
    fig = build_score_heatmap_figure(results, bin_size=bin_size)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_sweep(candles, min_score_range, use_regime_filter):
    sweep_results = []
    for score in min_score_range:
        result = simulate_backtest(candles, score, use_regime_filter)
        sweep_results.append({
            "min_score": score,
            "trade_count": result["total_trades"],
            "win_rate": result["win_rate"],
            "total_pnl": result["total_pnl"],
            "expectancy": result["expectancy"],
            "profit_factor": result["profit_factor"],
            "max_drawdown": result["max_drawdown"],
            "sharpe_ratio": result["sharpe_ratio"],
        })
    return sweep_results


def print_sweep_report(sweep_results):
    print("Minimum score sweep")
    print("-------------------")
    print(f"{'score':>5}  {'trades':>6}  {'win_rate':>8}  {'total_pnl':>10}  {'expectancy':>10}  {'profit_factor':>13}  {'max_dd':>10}  {'sharpe':>7}")
    for row in sweep_results:
        print(
            f"{row['min_score']:5d}  {row['trade_count']:6d}  "
            f"{row['win_rate']:8.2f}%  {row['total_pnl']:+10.2f}  {row['expectancy']:+10.2f}  "
            f"{row['profit_factor']:12.2f}  {row['max_drawdown']:+10.2f}  {row['sharpe_ratio']:7.2f}"
        )


def export_grid_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fig_to_base64(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("ascii")
    plt.close(fig)
    return encoded


def generate_html_report(results, output_path: Path):
    score_fig = build_score_pnl_figure(results)
    equity_fig = build_equity_curve_figure(results)
    heatmap_fig = build_score_heatmap_figure(results)
    score_img = fig_to_base64(score_fig)
    equity_img = fig_to_base64(equity_fig)
    heatmap_img = fig_to_base64(heatmap_fig)

    summary_html = f"""
    <h1>MT5 Forex Strategy Report</h1>
    <p><strong>Total trades:</strong> {results['total_trades']}</p>
    <p><strong>Win rate:</strong> {results['win_rate']:.2f}%</p>
    <p><strong>Net P&L:</strong> ${results['total_pnl']:+,.2f}</p>
    <p><strong>Average win:</strong> ${results['average_win']:+,.2f}</p>
    <p><strong>Average loss:</strong> ${results['average_loss']:+,.2f}</p>
    <p><strong>Profit factor:</strong> {results['profit_factor']:.2f}</p>
    <p><strong>Expectancy:</strong> ${results['expectancy']:+,.2f}</p>
    <p><strong>Max drawdown:</strong> ${results['max_drawdown']:+,.2f}</p>
    <p><strong>Sharpe ratio:</strong> {results['sharpe_ratio']:.2f}</p>
    """

    regime_rows = "".join(
        f"<tr><td>{html.escape(regime)}</td><td>{count}</td></tr>"
        for regime, count in results["regimes"].most_common(12)
    )
    reason_rows = "".join(
        f"<tr><td>{html.escape(reason)}</td><td>{count}</td></tr>"
        for reason, count in results["reasons"].most_common(12)
    )

    html_content = f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <title>MT5 Forex Strategy Report</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 24px; }}
          h1 {{ margin-bottom: 0.5rem; }}
          img {{ max-width: 100%; height: auto; border: 1px solid #ccc; margin-bottom: 24px; }}
          table {{ border-collapse: collapse; margin-bottom: 24px; width: 100%; max-width: 800px; }}
          th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
          th {{ background: #f3f3f3; }}
        </style>
      </head>
      <body>
        {summary_html}
        <h2>Equity Curve</h2>
        <img src="data:image/png;base64,{equity_img}" alt="Equity Curve">
        <h2>Score vs P&L</h2>
        <img src="data:image/png;base64,{score_img}" alt="Score vs PnL">
        <h2>Score/Outcome Heatmap</h2>
        <img src="data:image/png;base64,{heatmap_img}" alt="Score Outcome Heatmap">
        <h2>Top regimes</h2>
        <table>
          <thead><tr><th>Regime</th><th>Count</th></tr></thead>
          <tbody>{regime_rows}</tbody>
        </table>
        <h2>Top reasons</h2>
        <table>
          <thead><tr><th>Reason</th><th>Count</th></tr></thead>
          <tbody>{reason_rows}</tbody>
        </table>
      </body>
    </html>
    """

    output_path.write_text(html_content, encoding="utf-8")


def recommend_parameters(candles, min_score_range, metric, min_trades, compare_regime):
    candidates = []
    for score in min_score_range:
        for regime_flag in ([False, True] if compare_regime else [False]):
            result = simulate_backtest(candles, score, regime_flag)
            if result["total_trades"] < min_trades:
                continue
            candidates.append({
                "min_score": score,
                "regime_filter": regime_flag,
                "total_trades": result["total_trades"],
                "win_rate": result["win_rate"],
                "total_pnl": result["total_pnl"],
                "expectancy": result["expectancy"],
                "profit_factor": result["profit_factor"],
                "max_drawdown": result["max_drawdown"],
                "sharpe_ratio": result["sharpe_ratio"],
            })
    if not candidates:
        raise ValueError("No candidate parameters met the minimum trade threshold.")
    candidates.sort(key=lambda x: (x[metric], x["total_pnl"] if metric != "total_pnl" else x[metric]), reverse=True)
    return candidates


def print_recommendation(candidate_list, metric, top):
    print(f"Best parameter recommendations by {metric}")
    print("---------------------------------------")
    print(f"{'score':>5}  {'regime':>12}  {'trades':>6}  {'win_rate':>8}  {'total_pnl':>10}  {'expectancy':>10}  {'pf':>7}  {'max_dd':>10}  {'sharpe':>7}")
    for row in candidate_list[:top]:
        print(
            f"{row['min_score']:5d}  {str(row['regime_filter']):>12s}  {row['total_trades']:6d}  "
            f"{row['win_rate']:8.2f}%  {row['total_pnl']:+10.2f}  {row['expectancy']:+10.2f}  "
            f"{row['profit_factor']:7.2f}  {row['max_drawdown']:+10.2f}  {row['sharpe_ratio']:7.2f}"
        )
    best = candidate_list[0]
    print("\nRecommended best parameters:")
    print(f"  minimum_score = {best['min_score']}")
    print(f"  regime_filter = {best['regime_filter']}")
    print(f"  metric = {metric}")


def run_backtest(candles, min_score, use_regime_filter):
    results = simulate_backtest(candles, min_score, use_regime_filter)
    print("=======================================")
    print("  MT5 Forex Backtest Summary")
    print("=======================================")
    print(f"  Trades simulated:  {results['total_trades']}")
    print(f"  Wins: {results['wins']}  |  Losses: {results['losses']}")
    print(f"  Win rate:          {results['win_rate']:.2f}%")
    print(f"  Net P&L:           {results['total_pnl']:+,.2f}")
    print(f"  Avg win:           {results['average_win']:+,.2f}")
    print(f"  Avg loss:          {results['average_loss']:+,.2f}")
    print(f"  Profit factor:     {results['profit_factor']:.2f}")
    print(f"  Expectancy:        {results['expectancy']:+,.2f}")
    print(f"  Max drawdown:      {results['max_drawdown']:+,.2f}")
    print(f"  Sharpe ratio:      {results['sharpe_ratio']:.2f}")
    print(f"  Avg score:         {results['average_score']:.1f}")
    print("---------------------------------------")
    print("Direction counts:")
    for direction, count in results["direction_counts"].items():
        print(f"  {direction}: {count}")
    print("Regime counts:")
    for regime, count in results["regimes"].most_common(12):
        print(f"  {regime}: {count}")
    print("Top reasons:")
    for reason, count in results["reasons"].most_common(12):
        print(f"  {reason}: {count}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Advanced MT5 Forex strategy evaluation and reporting.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_backtest = subparsers.add_parser("backtest", help="Run a single backtest")
    parser_backtest.add_argument("file")
    parser_backtest.add_argument("--min-score", type=int, default=SETTINGS.minimum_score)
    parser_backtest.add_argument("--regime-filter", action="store_true")

    parser_sweep = subparsers.add_parser("sweep", help="Sweep minimum_score values")
    parser_sweep.add_argument("file")
    parser_sweep.add_argument("--min-score", type=int, nargs=3, default=[40, 90, 5],
                              metavar=("START", "STOP", "STEP"))
    parser_sweep.add_argument("--regime-filter", action="store_true")
    parser_sweep.add_argument("--csv", default=None, help="Export sweep results to CSV")

    parser_report = subparsers.add_parser("report", help="Generate a strategy chart report")
    parser_report.add_argument("file")
    parser_report.add_argument("--output-prefix", default="strategy_report")
    parser_report.add_argument("--min-score", type=int, default=SETTINGS.minimum_score)
    parser_report.add_argument("--regime-filter", action="store_true")
    parser_report.add_argument("--html-report", default=None, help="Save a combined HTML report with embedded charts")

    parser_recommend = subparsers.add_parser("recommend", help="Recommend best parameters automatically")
    parser_recommend.add_argument("file")
    parser_recommend.add_argument("--min-score", type=int, nargs=3, default=[40, 90, 5],
                                metavar=("START", "STOP", "STEP"))
    parser_recommend.add_argument("--metric", choices=["total_pnl", "expectancy", "profit_factor", "win_rate", "sharpe_ratio"], default="expectancy")
    parser_recommend.add_argument("--min-trades", type=int, default=10)
    parser_recommend.add_argument("--compare-regime", action="store_true", help="Evaluate both regime filter off and on")
    parser_recommend.add_argument("--top", type=int, default=3)
    parser_recommend.add_argument("--csv", default=None, help="Export recommendation grid to CSV")

    args = parser.parse_args()
    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    candles = load_historical_candles(path)

    if args.command == "backtest":
        run_backtest(candles, args.min_score, args.regime_filter)
    elif args.command == "sweep":
        start, stop, step = args.min_score
        sweep_results = run_sweep(candles, range(start, stop + 1, step), args.regime_filter)
        print_sweep_report(sweep_results)
        if args.csv:
            export_grid_csv(Path(args.csv), sweep_results, [
                "min_score", "trade_count", "win_rate", "total_pnl", "expectancy", "profit_factor", "max_drawdown", "sharpe_ratio"
            ])
            print(f"Sweep results exported to {args.csv}")
    elif args.command == "report":
        results = run_backtest(candles, args.min_score, args.regime_filter)
        prefix = args.output_prefix
        plot_pnl_vs_score(results, Path(f"{prefix}_score_pnl.png"))
        plot_equity_curve(results, Path(f"{prefix}_equity_curve.png"))
        plot_score_heatmap(results, Path(f"{prefix}_score_heatmap.png"))
        print(f"Charts saved to {prefix}_score_pnl.png, {prefix}_equity_curve.png, {prefix}_score_heatmap.png")
        if args.html_report:
            generate_html_report(results, Path(args.html_report))
            print(f"HTML report saved to {args.html_report}")
    elif args.command == "recommend":
        min_start, min_stop, min_step = args.min_score
        recommendations = recommend_parameters(
            candles,
            range(min_start, min_stop + 1, min_step),
            args.metric,
            args.min_trades,
            args.compare_regime,
        )
        print_recommendation(recommendations, args.metric, args.top)
        if args.csv:
            export_grid_csv(Path(args.csv), recommendations, [
                "min_score", "regime_filter", "total_trades", "win_rate", "total_pnl", "expectancy", "profit_factor", "max_drawdown", "sharpe_ratio"
            ])
            print(f"Recommendation grid exported to {args.csv}")


if __name__ == "__main__":
    main()
