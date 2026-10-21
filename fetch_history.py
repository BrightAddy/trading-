import argparse
import csv
import logging
from pathlib import Path
import MetaTrader5 as mt5

from config import SETTINGS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("mt5-fetcher")

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

def connect_mt5():
    if SETTINGS.mt5_login and SETTINGS.mt5_password:
        connected = mt5.initialize(
            login=SETTINGS.mt5_login,
            password=SETTINGS.mt5_password,
            server=SETTINGS.mt5_server
        )
    else:
        connected = mt5.initialize()

    if not connected:
        err_code, err_msg = mt5.last_error()
        raise ConnectionError(f"Failed to connect to MT5 ({err_code}: {err_msg}). Ensure MT5 desktop is running.")
    log.info("Successfully connected to MetaTrader 5.")

_SYMBOL_CACHE: dict[str, str] = {}

def resolve_symbol(raw_symbol: str):
    if raw_symbol in _SYMBOL_CACHE:
        cached = _SYMBOL_CACHE[raw_symbol]
        if mt5.symbol_select(cached, True):
            return cached

    clean = raw_symbol.upper().replace("/", "").replace("-", "").strip()

    for candidate in (raw_symbol, raw_symbol.upper(), raw_symbol.lower()):
        info = mt5.symbol_info(candidate)
        if info and mt5.symbol_select(candidate, True):
            _SYMBOL_CACHE[raw_symbol] = candidate
            return candidate

    all_symbols = mt5.symbols_get()
    if all_symbols:
        for s in all_symbols:
            if s.name.upper() == clean:
                if mt5.symbol_select(s.name, True):
                    _SYMBOL_CACHE[raw_symbol] = s.name
                    return s.name

        for s in all_symbols:
            if clean in s.name.upper():
                if mt5.symbol_select(s.name, True):
                    _SYMBOL_CACHE[raw_symbol] = s.name
                    return s.name

    log.warning(f"Failed to select {raw_symbol} in Market Watch")
    return None

def fetch_candles_for_asset(symbol: str, count: int, timeframe_str: str = "M1"):
    resolved = resolve_symbol(symbol)
    if not resolved:
        return []

    tf = TIMEFRAME_MAP.get(timeframe_str, mt5.TIMEFRAME_M1)
    log.info(f"Fetching {count} candles for {resolved} ({timeframe_str})...")
    rates = mt5.copy_rates_from_pos(resolved, tf, 0, count)
    if rates is None:
        log.warning(f"No candles retrieved for {symbol}")
        return []

    return [
        {
            "timestamp": int(r["time"]),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": float(r["tick_volume"]),
        }
        for r in rates
    ]

def save_candles_to_csv(candles, output_path: Path):
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for c in candles:
            writer.writerow([
                c["timestamp"],
                c["open"],
                c["high"],
                c["low"],
                c["close"],
                c["volume"]
            ])
    log.info(f"Saved {len(candles)} candles to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Fetch historical candle data from MetaTrader 5.")
    parser.add_argument("--asset", default="EURUSD", help="Asset symbol to fetch (default: EURUSD)")
    parser.add_argument("--candles", type=int, default=500, help="Number of candles to fetch (default: 500)")
    parser.add_argument("--timeframe", default=SETTINGS.timeframe, help="Candle timeframe (M1, M5, M15, etc.)")
    parser.add_argument("--output", default="history.csv", help="Output CSV file path (default: history.csv)")
    args = parser.parse_args()

    SETTINGS.validate()
    connect_mt5()
    try:
        candles = fetch_candles_for_asset(args.asset, args.candles, args.timeframe)
        if candles:
            save_candles_to_csv(candles, Path(args.output))
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()

