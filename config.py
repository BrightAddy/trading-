import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass(frozen=True)
class Settings:
    # Strategy Selector: "mtf_pullback", "session_breakout", or "legacy"
    strategy: str = os.getenv("TRADING_STRATEGY", "mtf_pullback")

    # MT5 Credentials (optional if MT5 terminal is already logged in on PC)
    mt5_login: int = int(os.getenv("MT5_LOGIN", "0")) if os.getenv("MT5_LOGIN", "0").isdigit() else 0
    mt5_password: str = os.getenv("MT5_PASSWORD", "")
    mt5_server: str = os.getenv("MT5_SERVER", "MetaQuotes-Demo")
    
    # Forex Trading Assets
    assets: tuple[str, ...] = (
        "EURUSD", "GBPUSD", "USDJPY", "AUDCAD", "EURCAD",
        "NZDCAD", "AUDCHF", "EURGBP", "USDCHF",
    )
    
    # Candle / Timeframe Settings
    timeframe: str = "M5"  # Primary execution timeframe
    candle_count: int = 240
    
    # ── Strategy 1: Multi-Timeframe Pullback Settings ─────────────
    mtf_higher_tf: str = "H1"             # Macro trend timeframe
    mtf_entry_tf: str = "M5"               # Pullback entry timeframe
    mtf_pullback_rsi_bull: float = 42.0    # RSI <= 42 triggers discount buy consideration
    mtf_pullback_rsi_bear: float = 58.0    # RSI >= 58 triggers premium sell consideration
    
    # ── Strategy 2: Session Volatility Breakout Settings ──────────
    session_asian_start: str = "00:00"     # UTC
    session_asian_end: str = "07:00"       # UTC
    session_london_start: str = "07:00"    # UTC
    session_london_end: str = "11:00"      # UTC
    session_ny_start: str = "12:30"        # UTC
    session_ny_end: str = "16:30"          # UTC
    session_breakout_buffer_pips: float = 2.5
    
    # ── Position Sizing & Forex Risk Management ───────────────────
    lot_size: float = 0.01               # Minimum micro-lot size
    enable_dynamic_sizing: bool = True   # Automatically calculate lot size from balance
    risk_per_trade_pct: float = 0.01     # 1% risk per trade
    
    # ATR Multipliers for Stop Loss & Take Profit
    sl_atr_multiplier: float = 1.5       # Stop Loss = 1.5 * ATR
    tp_atr_multiplier: float = 2.0       # Take Profit = 2.0 * ATR (Achievable intraday target)
    
    # ── Risk Shield & Execution Rules ─────────────────────────────
    minimum_score: int = 80              # Score threshold for legacy / composite gates
    max_daily_loss: float = float(os.getenv("MAX_DAILY_LOSS", "100.0"))
    daily_profit_target: float = float(os.getenv("DAILY_PROFIT_TARGET", "100.0"))
    max_trades_daily: int = 20
    max_consecutive_losses: int = 3      # Circuit breaker trips after 3 consecutive losses
    cooldown_minutes: float = 30.0       # Symbol locked for 30 mins after any trade exit
    scan_delay_seconds: int = 10
    max_open_positions: int = 3
    max_positions_per_asset: int = 1     # Diversification: max 1 active trade per pair
    
    # Trailing Stop & Break-Even
    enable_break_even: bool = True       # Move SL to entry when trade moves into profit
    break_even_atr_trigger: float = 1.5  # Move to BE only at 1.5x ATR (gives room to breathe)
    enable_trailing_stop: bool = True    # Trail SL behind price once in strong profit
    trailing_stop_atr: float = 1.0       # Trail by 1.0 * ATR
    
    slippage_points: int = 20
    magic_number: int = 100200
    
    # Backtesting Settings
    backtest_candle_horizon: int = 40
    backtest_lot_size: float = 0.05
    
    # Invo Copy Trading Settings
    invo_poll_interval_seconds: float = 3.0
    invo_target_traders: tuple[str, ...] = tuple(
        t.strip() for t in os.getenv("INVO_TARGET_TRADERS", "").split(",") if t.strip()
    )
    invo_symbol_map: dict = None
    invo_close_on_exit: bool = True
    invo_copy_sl_tp: bool = True
    
    log_file: str = "trade_log.csv"
    
    def validate(self):
        if self.lot_size <= 0:
            raise ValueError("Lot size must be positive.")
        if self.max_daily_loss <= 0:
            raise ValueError("Max daily loss must be positive.")
        if self.strategy not in ("mtf_pullback", "session_breakout", "legacy"):
            raise ValueError(f"Unknown strategy: {self.strategy}. Use 'mtf_pullback', 'session_breakout', or 'legacy'.")

SETTINGS = Settings()
