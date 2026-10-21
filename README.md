# MetaTrader 5 (MT5) Trading Engine

An automated rule-based Forex trading engine for **MetaTrader 5 (MT5)**. It connects directly to your MT5 desktop terminal via the official `MetaTrader5` Python package for live scanning, backtesting, paper trading (DRY-RUN), and automated market execution.

## Quick Setup (Windows)

1. **MetaTrader 5 Terminal:** Open your **MetaTrader 5** desktop terminal on your PC and log into your account (Demo or Live).
2. Open terminal in this folder and activate virtual environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` (Optional: specify `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER` if MT5 terminal isn't already open).

## Running the Engine

- **Dry-Run Mode (Paper Trading simulation):**
  ```powershell
  python main.py run --dry-run
  ```
- **Live MT5 Trading Mode:**
  ```powershell
  python main.py run
  ```
- **Fetch Historical Market Data from MT5:**
  ```powershell
  python main.py fetch --asset EURUSD --candles 1000 --timeframe M5
  ```
- **Backtest Strategy:**
  ```powershell
  python main.py backtest history.csv --regime-filter
  ```
- **Invo App to MT5 Copy Trading (Dry-Run):**
  ```powershell
  python main.py copy --trader <trader_username> --dry-run
  ```
- **Live Copy Trading from Invo App to MT5:**
  ```powershell
  python main.py copy --trader <trader_username> --risk-pct 0.01
  ```

## Key Features

- **Invo App Copy Trading:** Automatically syncs and copies open positions and closes from top Invo traders directly to your MT5 account.
- Multi-asset scanning (`EURUSD`, `GBPUSD`, `USDJPY`, `AUDCAD`, `BTCUSD`, etc.)
- Dynamic ATR-based Stop Loss (SL) & Take Profit (TP)
- Dynamic Lot Sizing based on account balance & risk percentage
- Maximum concurrent positions limit & daily loss guard

