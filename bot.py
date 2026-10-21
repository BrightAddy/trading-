import argparse
import csv
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import MetaTrader5 as mt5

from config import SETTINGS
from regime import regime_filter, signal_matches_regime
from risk import RiskManager
from strategy import (
    analyse,
    analyse_mtf_pullback,
    analyse_session_breakout,
    atr,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("mt5-bot")

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
    """Initialize connection to MetaTrader 5 terminal."""
    if SETTINGS.mt5_login and SETTINGS.mt5_password:
        connected = mt5.initialize(
            login=SETTINGS.mt5_login,
            password=SETTINGS.mt5_password,
            server=SETTINGS.mt5_server
        )
    else:
        connected = mt5.initialize()

    if not connected:
        error_code, error_msg = mt5.last_error()
        raise ConnectionError(f"Failed to connect to MetaTrader 5 (Error {error_code}: {error_msg}). Ensure MT5 desktop is open.")

    account = mt5.account_info()
    if account is None:
        raise ConnectionError("Connected to MT5, but could not retrieve account info.")

    log.info(
        "✅ Connected to MT5 | Account: %s (%s) | Server: %s | Balance: %.2f %s",
        account.login, account.name, account.server, account.balance, account.currency
    )
    return account

_SYMBOL_CACHE: dict[str, str] = {}

def resolve_symbol(raw_symbol: str) -> Optional[str]:
    """Resolve raw symbol to broker's exact MT5 symbol name (handling casing, prefixes, suffixes)."""
    if raw_symbol in _SYMBOL_CACHE:
        cached = _SYMBOL_CACHE[raw_symbol]
        if mt5.symbol_select(cached, True):
            return cached

    clean = raw_symbol.upper().replace("/", "").replace("-", "").strip()

    # 1. Direct match or upper/lower
    for candidate in (raw_symbol, raw_symbol.upper(), raw_symbol.lower()):
        info = mt5.symbol_info(candidate)
        if info and mt5.symbol_select(candidate, True):
            _SYMBOL_CACHE[raw_symbol] = candidate
            return candidate

    # 2. Search broker's available symbols list
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

    log.warning("Failed to select symbol %s in Market Watch", raw_symbol)
    return None

def get_candles(symbol: str, timeframe_str: str = "M5", count: int = 240):
    """Retrieve historical candle bars from MT5."""
    resolved = resolve_symbol(symbol)
    if not resolved:
        return None

    tf = TIMEFRAME_MAP.get(timeframe_str, mt5.TIMEFRAME_M5)
    rates = mt5.copy_rates_from_pos(resolved, tf, 0, count)
    if rates is None or len(rates) < min(count, 30):
        return None

    return [
        {
            "time": int(r["time"]),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": float(r["tick_volume"]),
        }
        for r in rates
    ]

def append_log(asset, direction, score, regime, lots, entry_price, sl, tp, ticket, reasons, mode="DEMO"):
    """Log trades to CSV."""
    path = Path(SETTINGS.log_file)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([
                "timestamp", "mode", "asset", "direction", "score",
                "regime", "lots", "entry_price", "stop_loss", "take_profit", "ticket", "reasons"
            ])
        w.writerow([
            datetime.now().isoformat(), mode, asset, direction, score,
            regime, lots, entry_price, sl, tp, ticket, "|".join(reasons)
        ])

def calculate_lot_size(account: Optional[mt5.AccountInfo], symbol: str, sl_pips: float, order_type: int = mt5.ORDER_TYPE_BUY, price: float = 0.0) -> float:
    """Calculate lot size based on fixed or risk-percentage sizing with margin safety."""
    if not SETTINGS.enable_dynamic_sizing:
        return SETTINGS.lot_size

    info = mt5.symbol_info(symbol)
    if not info:
        return SETTINGS.lot_size

    min_lot = info.volume_min or 0.01
    max_lot = min(info.volume_max or 10.0, 5.0)  # Safe cap to prevent excessive margin usage
    lot_step = info.volume_step or 0.01

    balance = account.balance if account else 10000.0
    free_margin = account.margin_free if account else balance

    # Risk amount (e.g. 1% of balance)
    risk_amount = balance * SETTINGS.risk_per_trade_pct

    # Pip value calculation per standard lot
    tick_value = getattr(info, 'trade_tick_value', 0.0) or 10.0
    tick_size = getattr(info, 'trade_tick_size', 0.0) or info.point
    point = info.point or 0.0001
    pip_points = 10 * point if info.digits in (3, 5) else point
    pip_value_per_lot = (tick_value / max(tick_size, 1e-6)) * pip_points
    if pip_value_per_lot <= 0:
        pip_value_per_lot = 10.0

    calculated_lots = risk_amount / (max(sl_pips, 1.0) * pip_value_per_lot)

    # Margin check: ensure required margin does not exceed 20% of free margin
    if price > 0:
        margin_1lot = mt5.order_calc_margin(order_type, symbol, 1.0, price)
        if margin_1lot and margin_1lot > 0:
            max_lots_by_margin = (free_margin * 0.20) / margin_1lot
            calculated_lots = min(calculated_lots, max_lots_by_margin)

    lots = max(min_lot, min(calculated_lots, max_lot))
    steps = round(lots / lot_step)
    return max(min_lot, round(steps * lot_step, 2))

def execute_order(symbol: str, direction: str, score: int, regime: str, reasons: list[str], dry_run: bool = False):
    """Place a market order on MT5 with asymmetric ATR-based SL and TP (1:2 R:R)."""
    resolved = resolve_symbol(symbol)
    if not resolved:
        log.warning("Cannot trade %s: symbol not found or available", symbol)
        return False, 0

    info = mt5.symbol_info(resolved)
    if not info:
        log.warning("Cannot get info for %s", resolved)
        return False, 0

    tick = mt5.symbol_info_tick(resolved)
    if not tick:
        log.warning("Cannot get price tick for %s", resolved)
        return False, 0

    rates = get_candles(symbol, SETTINGS.timeframe, 30)
    if not rates:
        log.warning("Cannot fetch candle history for %s to compute ATR Stop-Loss", symbol)
        return False, 0

    highs = [c["high"] for c in rates]
    lows = [c["low"] for c in rates]
    closes = [c["close"] for c in rates]
    atr_val = atr(highs, lows, closes, 14)
    if atr_val <= 0:
        atr_val = (max(highs) - min(lows)) / 14

    digits = info.digits
    point = info.point
    sl_distance = max(atr_val * SETTINGS.sl_atr_multiplier, 10 * point)
    tp_distance = max(atr_val * SETTINGS.tp_atr_multiplier, 20 * point)

    if direction == "call":
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
        sl = round(price - sl_distance, digits)
        tp = round(price + tp_distance, digits)
        action_name = "BUY"
    elif direction == "put":
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        sl = round(price + sl_distance, digits)
        tp = round(price - tp_distance, digits)
        action_name = "SELL"
    else:
        return False, 0

    account = mt5.account_info()
    sl_pips = sl_distance / max(point * 10 if digits in (3, 5) else point, 1e-6)
    lots = calculate_lot_size(account, resolved, sl_pips, order_type=order_type, price=price)

    if dry_run:
        ticket = int(time.time())
        log.info(
            "🧪 [DRY-RUN] %s %s | Lots: %.2f | Price: %.*f | SL: %.*f | TP: %.*f | Score: %d",
            action_name, resolved, lots, digits, price, digits, sl, digits, tp, score
        )
        append_log(resolved, action_name, score, regime, lots, price, sl, tp, ticket, reasons, mode="DRY_RUN")
        return True, ticket

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": resolved,
        "volume": lots,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": SETTINGS.slippage_points,
        "magic": SETTINGS.magic_number,
        "type_time": mt5.ORDER_TIME_GTC,
    }

    filling_modes = [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_FOK]
    result = None
    for filling in filling_modes:
        request["type_filling"] = filling
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            break
        if result and result.retcode not in (mt5.TRADE_RETCODE_INVALID_FILL, 10030):
            break

    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(
            "🚀 [ORDER FILLED] %s %s | Lots: %.2f | Price: %.*f | SL: %.*f | TP: %.*f | Ticket: %d",
            action_name, symbol, lots, digits, result.price, digits, sl, digits, tp, result.order
        )
        append_log(symbol, action_name, score, regime, lots, result.price, sl, tp, result.order, reasons, mode="LIVE")
        return True, result.order
    else:
        retcode = result.retcode if result else "None"
        comment = result.comment if result else mt5.last_error()
        log.warning("❌ Order rejected for %s (%s): %s (%s)", symbol, action_name, retcode, comment)
        return False, 0

def manage_open_positions(open_positions):
    """
    Two-Stage Trade Management:
    1. Break-Even: Move SL to entry + 2 pips once price reaches 1.5 * ATR profit.
    2. Dynamic Trailing Stop: Trail SL behind price by 1.0 * ATR once profit exceeds 2.0 * ATR.
    """
    if not open_positions:
        return

    for pos in open_positions:
        try:
            symbol = pos.symbol
            info = mt5.symbol_info(symbol)
            if not info:
                continue

            entry = pos.price_open
            current_sl = pos.sl
            tp = pos.tp
            pos_type = pos.type  # 0: BUY, 1: SELL
            digits = info.digits
            point = info.point

            rates = get_candles(symbol, SETTINGS.timeframe, 30)
            if not rates:
                continue

            highs = [c["high"] for c in rates]
            lows = [c["low"] for c in rates]
            closes = [c["close"] for c in rates]
            atr_val = atr(highs, lows, closes, 14)
            if atr_val <= 0:
                continue

            be_distance = atr_val * getattr(SETTINGS, "break_even_atr_trigger", 1.5)
            trail_distance = atr_val * getattr(SETTINGS, "trailing_stop_atr", 1.0)
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                continue

            pip_mult = 10 if digits in (3, 5) else 1

            # ── BUY POSITIONS ─────────────────────────────────────────
            if pos_type == mt5.ORDER_TYPE_BUY:
                current_price = tick.bid
                profit_distance = current_price - entry

                # Stage 2: Dynamic Trailing Stop (when profit >= 2.0 * ATR)
                if getattr(SETTINGS, "enable_trailing_stop", True) and profit_distance >= (atr_val * 2.0):
                    trail_sl = round(current_price - trail_distance, digits)
                    if trail_sl > current_sl and trail_sl > entry:
                        req = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": pos.ticket,
                            "symbol": symbol,
                            "sl": trail_sl,
                            "tp": tp,
                        }
                        res = mt5.order_send(req)
                        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                            log.info("📈 [TRAILING STOP UPDATED] %s BUY Ticket %d: SL trailed to %.5f", symbol, pos.ticket, trail_sl)

                # Stage 1: Break-Even (when profit >= 1.5 * ATR)
                elif getattr(SETTINGS, "enable_break_even", True) and profit_distance >= be_distance:
                    if current_sl < entry or current_sl == 0:
                        new_sl = round(entry + (2 * point * pip_mult), digits)
                        req = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": pos.ticket,
                            "symbol": symbol,
                            "sl": new_sl,
                            "tp": tp,
                        }
                        res = mt5.order_send(req)
                        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                            log.info("🛡️ [BREAK-EVEN ACTIVATED] %s BUY Ticket %d: SL moved to %.5f (Risk-Free)", symbol, pos.ticket, new_sl)

            # ── SELL POSITIONS ────────────────────────────────────────
            elif pos_type == mt5.ORDER_TYPE_SELL:
                current_price = tick.ask
                profit_distance = entry - current_price

                # Stage 2: Dynamic Trailing Stop
                if getattr(SETTINGS, "enable_trailing_stop", True) and profit_distance >= (atr_val * 2.0):
                    trail_sl = round(current_price + trail_distance, digits)
                    if (current_sl == 0 or trail_sl < current_sl) and trail_sl < entry:
                        req = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": pos.ticket,
                            "symbol": symbol,
                            "sl": trail_sl,
                            "tp": tp,
                        }
                        res = mt5.order_send(req)
                        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                            log.info("📉 [TRAILING STOP UPDATED] %s SELL Ticket %d: SL trailed to %.5f", symbol, pos.ticket, trail_sl)

                # Stage 1: Break-Even
                elif getattr(SETTINGS, "enable_break_even", True) and profit_distance >= be_distance:
                    if current_sl > entry or current_sl == 0:
                        new_sl = round(entry - (2 * point * pip_mult), digits)
                        req = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": pos.ticket,
                            "symbol": symbol,
                            "sl": new_sl,
                            "tp": tp,
                        }
                        res = mt5.order_send(req)
                        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                            log.info("🛡️ [BREAK-EVEN ACTIVATED] %s SELL Ticket %d: SL moved to %.5f (Risk-Free)", symbol, pos.ticket, new_sl)
        except Exception as e:
            log.debug("Error managing position %s: %s", pos.ticket, e)

def run_bot(strategy: Optional[str] = None, dry_run: bool = False, headless: bool = False, max_trades_limit: int = None):
    """Main scanning and trading execution loop supporting all 3 strategies & unified Risk Shield."""
    active_strategy = strategy or SETTINGS.strategy
    SETTINGS.validate()
    account = connect_mt5()
    
    risk = RiskManager(
        max_loss=SETTINGS.max_daily_loss,
        max_trades=max_trades_limit or SETTINGS.max_trades_daily,
        max_consecutive_losses=SETTINGS.max_consecutive_losses,
        daily_profit_target=SETTINGS.daily_profit_target,
        cooldown_minutes=SETTINGS.cooldown_minutes,
    )

    trades_executed = 0
    log.info("=" * 65)
    log.info("🤖 UNIFIED TRADING ENGINE ACTIVE")
    log.info("⚡ Strategy: %s", active_strategy.upper())
    log.info("🛡️ Risk Shield: Cooldown=%dm | MaxConsLoss=%d | 1:2 R:R", SETTINGS.cooldown_minutes, SETTINGS.max_consecutive_losses)
    log.info("📊 Assets: %d Forex pairs (%s)", len(SETTINGS.assets), ", ".join(SETTINGS.assets[:4]) + "...")
    log.info("=" * 65)

    try:
        while True:
            if max_trades_limit and trades_executed >= max_trades_limit:
                log.info("Reached session trade limit (%d). Stopping cleanly.", max_trades_limit)
                break

            # 1. Synchronize live closed deals from MT5 history to update daily P&L and trigger cooldowns
            risk.sync_with_mt5_history()

            # 2. Check risk manager circuit breakers
            trade_risk = (account.balance * SETTINGS.risk_per_trade_pct) if account else 10.0
            allowed, risk_reason = risk.allowed(trade_risk)
            if not allowed:
                log.warning("🚨 [RISK MANAGER HALTED] %s. Pausing bot loop.", risk_reason)
                time.sleep(SETTINGS.scan_delay_seconds * 2)
                continue

            # 3. Manage open positions (Trailing Stop & Break-Even)
            open_positions = mt5.positions_get()
            open_count = len(open_positions) if open_positions is not None else 0

            if open_positions:
                manage_open_positions(open_positions)

            if open_count >= SETTINGS.max_open_positions:
                log.info("Max concurrent positions reached (%d/%d). Managing active trades...", open_count, SETTINGS.max_open_positions)
                time.sleep(SETTINGS.scan_delay_seconds)
                continue

            active_symbols = set()
            if open_positions:
                for p in open_positions:
                    active_symbols.add(p.symbol.upper())

            # 4. Scan assets under active strategy
            ranked = []
            for asset in SETTINGS.assets:
                resolved = resolve_symbol(asset)
                if not resolved:
                    continue

                # Rule A: Prevent stacking (already open)
                if resolved.upper() in active_symbols:
                    continue

                # Rule B: Anti-reentry cooldown check (protect against whipsaws)
                is_locked, remaining_mins = risk.is_in_cooldown(resolved)
                if is_locked:
                    continue

                try:
                    # ── STRATEGY 1: Multi-Timeframe Pullback (H1 + M5) ────────
                    if active_strategy == "mtf_pullback":
                        h1_candles = get_candles(asset, SETTINGS.mtf_higher_tf, 80)
                        m5_candles = get_candles(asset, SETTINGS.mtf_entry_tf, 100)
                        if not m5_candles:
                            continue

                        signal = analyse_mtf_pullback(h1_candles, m5_candles)
                        regime = "mtf_pullback"
                        regime_ok = True

                    # ── STRATEGY 2: Session Volatility Breakout (London/NY) ───
                    elif active_strategy == "session_breakout":
                        candles = get_candles(asset, SETTINGS.timeframe, 150)
                        if not candles:
                            continue

                        signal = analyse_session_breakout(candles, buffer_pips=SETTINGS.session_breakout_buffer_pips)
                        regime = "session_breakout"
                        regime_ok = True

                    # ── STRATEGY 3: Legacy Multi-Gate System ─────────────────
                    else:
                        candles = get_candles(asset, SETTINGS.timeframe, SETTINGS.candle_count)
                        if not candles:
                            continue

                        signal = analyse(candles)
                        regime = regime_filter(candles)
                        regime_ok = signal_matches_regime(signal.direction, regime)

                    if signal.direction != "wait" and signal.score >= SETTINGS.minimum_score and regime_ok:
                        log.info(
                            "🎯 SIGNAL FOUND [%s]: %s | %s | Score: %d | Reasons: %s",
                            active_strategy.upper(), asset, signal.direction.upper(), signal.score, ",".join(signal.reasons)
                        )
                        ranked.append((signal.score, asset, signal, regime))

                except Exception as exc:
                    log.warning("Error scanning %s: %s", asset, exc)

            # 5. Execute highest conviction trade
            if ranked:
                _, best_asset, best_signal, best_regime = max(ranked, key=lambda x: x[0])
                ok, ticket = execute_order(
                    symbol=best_asset,
                    direction=best_signal.direction,
                    score=best_signal.score,
                    regime=best_regime,
                    reasons=list(best_signal.reasons),
                    dry_run=dry_run
                )
                if ok:
                    trades_executed += 1
            else:
                log.info("Scan completed (%s): no setups meeting score >= %d. Next scan in %ds...", active_strategy, SETTINGS.minimum_score, SETTINGS.scan_delay_seconds)

            time.sleep(SETTINGS.scan_delay_seconds)

    except KeyboardInterrupt:
        log.info("Bot manually stopped by user.")
    finally:
        mt5.shutdown()
        log.info("MT5 connection closed cleanly.")

def main():
    parser = argparse.ArgumentParser(description="Autonomous MetaTrader 5 Forex Trading Bot")
    parser.add_argument("--strategy", choices=["mtf_pullback", "session_breakout", "legacy"], default=None, help="Strategy to execute")
    parser.add_argument("--dry-run", action="store_true", help="Simulate trades without placing actual broker orders")
    parser.add_argument("--headless", action="store_true", help="Run without user prompts")
    parser.add_argument("--max-trades", type=int, default=None, help="Stop after N trades")
    args = parser.parse_args()

    run_bot(strategy=args.strategy, dry_run=args.dry_run, headless=args.headless, max_trades_limit=args.max_trades)

if __name__ == "__main__":
    main()
