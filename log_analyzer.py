from pathlib import Path

from utils import print_trade_summary, summarize_trade_log


def main():
    path = Path("trade_log.csv")
    summary = summarize_trade_log(path)
    print_trade_summary(summary)


if __name__ == "__main__":
    main()
