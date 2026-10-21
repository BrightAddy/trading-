import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="MetaTrader 5 Autonomous Trading Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  fetch       Download historical candle data from MetaTrader 5
  backtest    Run strategy backtest on historical CSV data
  sweep       Perform parameter grid sweeps over minimum score settings
  recommend   Search for optimal trading strategy parameters
  report      Generate visual PNG charts and interactive HTML performance report
  run         Launch the trading engine (supports DEMO, DRY-RUN, and LIVE MT5 modes)
  summary     Analyze trading logs in trade_log.csv

Examples:
  python main.py fetch --asset EURUSD --candles 500 --timeframe M5
  python main.py run --dry-run --max-trades 5
  python main.py backtest history.csv --regime-filter
  python main.py recommend history.csv --compare-regime
  python main.py report history.csv --html-report report.html
"""
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Fetch subcommand
    parser_fetch = subparsers.add_parser("fetch", help="Download historical candle data from MT5")
    parser_fetch.add_argument("--asset", default="EURUSD", help="Asset symbol (default: EURUSD)")
    parser_fetch.add_argument("--candles", type=int, default=500, help="Number of candles (default: 500)")
    parser_fetch.add_argument("--timeframe", default="M1", help="Candle timeframe: M1, M5, M15, etc. (default: M1)")
    parser_fetch.add_argument("--output", default="history.csv", help="Output file (default: history.csv)")

    # Backtest subcommand
    parser_bt = subparsers.add_parser("backtest", help="Run strategy backtest")
    parser_bt.add_argument("file", nargs="?", default="history.csv", help="Path to historical CSV")
    parser_bt.add_argument("--min-score", type=int, default=75)
    parser_bt.add_argument("--regime-filter", action="store_true")

    # Sweep subcommand
    parser_sw = subparsers.add_parser("sweep", help="Sweep minimum scores")
    parser_sw.add_argument("file", nargs="?", default="history.csv")
    parser_sw.add_argument("--min-score", type=int, nargs=3, default=[40, 90, 5])
    parser_sw.add_argument("--regime-filter", action="store_true")

    # Recommend subcommand
    parser_rec = subparsers.add_parser("recommend", help="Recommend optimal parameters")
    parser_rec.add_argument("file", nargs="?", default="history.csv")
    parser_rec.add_argument("--compare-regime", action="store_true")

    # Report subcommand
    parser_rep = subparsers.add_parser("report", help="Generate performance visual reports")
    parser_rep.add_argument("file", nargs="?", default="history.csv")
    parser_rep.add_argument("--html-report", default="report.html")

    # Run subcommand
    parser_run = subparsers.add_parser("run", help="Launch autonomous trading bot")
    parser_run.add_argument("--strategy", choices=["mtf_pullback", "session_breakout", "legacy"], default=None, help="Trading strategy (default: mtf_pullback)")
    parser_run.add_argument("--dry-run", action="store_true", help="Run paper trading mode")
    parser_run.add_argument("--headless", action="store_true", help="Non-interactive execution")
    parser_run.add_argument("--max-trades", type=int, default=None, help="Stop after N trades")

    # Copy subcommand (Invo App to MT5 Copier)
    parser_cp = subparsers.add_parser("copy", help="Copy trades from Invo App traders to MT5")
    parser_cp.add_argument("--trader", nargs="+", required=True, help="Invo trader username(s) or profile ID(s) to copy")
    parser_cp.add_argument("--amount", type=float, default=100.0, help="Dollar amount to trade per position (default: $100.00)")
    parser_cp.add_argument("--poll-interval", type=float, default=1.0, help="Polling interval in seconds (default: 1.0s)")
    parser_cp.add_argument("--dry-run", action="store_true", help="Simulate copying without sending live orders")
    parser_cp.add_argument("--lot-size", type=float, default=None, help="Fixed lot size (overrides dollar amount if set)")
    parser_cp.add_argument("--no-close-sync", action="store_true", help="Do not close MT5 trade when trader closes on Invo")

    # Summary subcommand
    parser_sum = subparsers.add_parser("summary", help="Analyze trading logs")
    parser_sum.add_argument("--log-file", default="trade_log.csv")

    args = parser.parse_args()

    if args.command == "fetch":
        import fetch_history
        sys.argv = ["fetch_history.py", "--asset", args.asset, "--candles", str(args.candles), "--timeframe", args.timeframe, "--output", args.output]
        fetch_history.main()

    elif args.command == "backtest":
        import backtester
        sys.argv = ["backtester.py", args.file]
        if args.regime_filter:
            sys.argv.append("--regime-filter")
        sys.argv.extend(["--min-score", str(args.min_score)])
        backtester.main()

    elif args.command == "sweep":
        import strategy_analysis
        sys.argv = ["strategy_analysis.py", "sweep", args.file, "--min-score"] + [str(x) for x in args.min_score]
        if args.regime_filter:
            sys.argv.append("--regime-filter")
        strategy_analysis.main()

    elif args.command == "recommend":
        import strategy_analysis
        sys.argv = ["strategy_analysis.py", "recommend", args.file]
        if args.compare_regime:
            sys.argv.append("--compare-regime")
        strategy_analysis.main()

    elif args.command == "report":
        import strategy_analysis
        sys.argv = ["strategy_analysis.py", "report", args.file, "--html-report", args.html_report]
        strategy_analysis.main()

    elif args.command == "run":
        import bot
        bot.run_bot(strategy=args.strategy, dry_run=args.dry_run, headless=args.headless, max_trades_limit=args.max_trades)

    elif args.command == "copy":
        from invo_copier import InvoMT5Copier
        copier = InvoMT5Copier(
            target_traders=args.trader,
            poll_interval=args.poll_interval,
            dry_run=args.dry_run,
            dollar_amount=args.amount,
            lot_size=args.lot_size,
            close_on_exit=not args.no_close_sync
        )
        copier.run()

    elif args.command == "summary":
        import utils
        path = Path(args.log_file)
        if not path.exists():
            print(f"Log file {path} does not exist yet. Run trades to generate logs.")
        else:
            summary = utils.summarize_trade_log(path)
            utils.print_trade_summary(summary)

if __name__ == "__main__":
    main()
