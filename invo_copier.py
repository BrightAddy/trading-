import argparse
import csv
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
import requests
import MetaTrader5 as mt5

from config import SETTINGS
from risk import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("invo-copier")

# Default mapping for common crypto perps to MT5 symbols
DEFAULT_SYMBOL_MAP = {
    "BTC": "BTCUSD",
    "BTC-PERP": "BTCUSD",
    "BTCUSDT": "BTCUSD",
    "BTCUSD": "BTCUSD",
    "ETH": "ETHUSD",
    "ETH-PERP": "ETHUSD",
    "ETHUSDT": "ETHUSD",
    "ETHUSD": "ETHUSD",
    "SOL": "SOLUSD",
    "SOL-PERP": "SOLUSD",
    "SOLUSDT": "SOLUSD",
    "SOLUSD": "SOLUSD",
    "XRP": "XRPUSD",
    "XRP-PERP": "XRPUSD",
    "XRPUSDT": "XRPUSD",
    "XRPUSD": "XRPUSD",
    "DOGE": "DOGEUSD",
    "DOGE-PERP": "DOGEUSD",
    "DOGEUSDT": "DOGEUSD",
    "DOGEUSD": "DOGEUSD",
}

class InvoPosition:
    def __init__(
        self,
        trade_id: str,
        trader: str,
        symbol: str,
        side: str,  # 'BUY' or 'SELL'
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: float = 1.0,
        timestamp: Optional[str] = None,
    ):
        self.trade_id = str(trade_id)
        self.trader = trader
        self.symbol = symbol.upper().strip()
        self.side = side.upper().strip()
        self.entry_price = float(entry_price)
        self.stop_loss = float(stop_loss) if stop_loss else None
        self.take_profit = float(take_profit) if take_profit else None
        self.leverage = float(leverage)
        self.timestamp = timestamp or datetime.now().isoformat()

    def __repr__(self):
        return f"<InvoPosition {self.trader} | {self.side} {self.symbol} @ {self.entry_price} | SL: {self.stop_loss} TP: {self.take_profit}>"


class InvoClient:
    """Client to query Invo platform data and track trader positions."""

    def __init__(self, base_url: str = "https://app.invoapp.com"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Origin": self.base_url,
            "Referer": f"{self.base_url}/",
        })

    def fetch_trader_positions(self, trader_identifier: str) -> List[InvoPosition]:
        """
        Poll active positions for a specific trader username or profile ID.
        Checks known Invo REST/GraphQL endpoints and falls back gracefully.
        """
        trader_clean = trader_identifier.strip().lstrip("@")
        positions: List[InvoPosition] = []

        # Common endpoints for social trading platforms
        candidate_endpoints = [
            f"{self.base_url}/api/traders/{trader_clean}/positions",
            f"{self.base_url}/api/v1/trader/{trader_clean}/trades",
            f"{self.base_url}/api/portfolio/{trader_clean}/open-orders",
            f"{self.base_url}/api/users/{trader_clean}/signals",
        ]

        for url in candidate_endpoints:
            try:
                resp = self.session.get(url, timeout=6)
                if resp.status_code == 200:
                    data = resp.json()
                    raw_list = data if isinstance(data, list) else data.get("positions", data.get("trades", data.get("data", [])))
                    for item in raw_list:
                        pos = self._parse_position(item, trader_clean)
                        if pos:
                            positions.append(pos)
                    if positions:
                        return positions
            except Exception:
                continue

        return positions

    def _parse_position(self, item: dict, trader: str) -> Optional[InvoPosition]:
        try:
            trade_id = item.get("id") or item.get("trade_id") or item.get("order_id") or f"{trader}_{item.get('symbol')}_{item.get('entry_price')}"
            symbol = item.get("symbol") or item.get("pair") or item.get("asset") or "BTCUSD"
            raw_side = (item.get("side") or item.get("direction") or item.get("type") or "BUY").upper()
            side = "BUY" if "BUY" in raw_side or "LONG" in raw_side or "CALL" in raw_side else "SELL"
            entry_price = float(item.get("entry_price") or item.get("price") or item.get("entry") or 0.0)
            sl = float(item["stop_loss"]) if item.get("stop_loss") else None
            tp = float(item["take_profit"]) if item.get("take_profit") else None
            leverage = float(item.get("leverage") or 1.0)

            return InvoPosition(
                trade_id=str(trade_id),
                trader=trader,
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                stop_loss=sl,
                take_profit=tp,
                leverage=leverage
            )
        except Exception as exc:
            log.debug("Could not parse position item: %s (%s)", item, exc)
            return None


class InvoMT5Copier:
    """Bridges Invo positions to MetaTrader 5 live/demo execution."""

    def __init__(
        self,
        target_traders: List[str],
        poll_interval: float = 1.0,
        dry_run: bool = False,
        dollar_amount: float = 100.0,
        lot_size: Optional[float] = None,
        risk_pct: Optional[float] = None,
        close_on_exit: bool = True,
        magic_number: int = 100300,
    ):
        self.target_traders = [t.strip().lstrip("@") for t in target_traders if t.strip()]
        self.poll_interval = poll_interval
        self.dry_run = dry_run
        self.dollar_amount = dollar_amount
        self.lot_size = lot_size
        self.risk_pct = risk_pct
        self.close_on_exit = close_on_exit
        self.magic_number = magic_number

        self.risk = RiskManager(
            max_loss=SETTINGS.max_daily_loss,
            max_trades=SETTINGS.max_trades_daily,
            max_consecutive_losses=SETTINGS.max_consecutive_losses,
            daily_profit_target=SETTINGS.daily_profit_target,
            cooldown_minutes=SETTINGS.cooldown_minutes,
        )

        self.client = InvoClient()
        self.active_copies: Dict[str, dict] = {}  # trade_id -> {'ticket': int, 'symbol': str, 'side': str}
        self.known_trader_positions: Set[str] = set()

    def connect_mt5(self):
        """Ensure connection to MetaTrader 5 terminal."""
        if SETTINGS.mt5_login and SETTINGS.mt5_password:
            connected = mt5.initialize(
                login=SETTINGS.mt5_login,
                password=SETTINGS.mt5_password,
                server=SETTINGS.mt5_server
            )
        else:
            connected = mt5.initialize()

        if not connected:
            code, msg = mt5.last_error()
            raise ConnectionError(f"Failed to connect to MetaTrader 5 (Error {code}: {msg}). Ensure MT5 desktop is open.")

        account = mt5.account_info()
        if account is None:
            raise ConnectionError("Connected to MT5, but could not read account info.")

        log.info(
            " Connected to MT5 | Account: %s (%s) | Balance: %.2f %s",
            account.login, account.name, account.balance, account.currency
        )
        return account

    def resolve_mt5_symbol(self, raw_symbol: str) -> Optional[str]:
        """Find the exact matching symbol in MT5 Market Watch."""
        clean = raw_symbol.upper().replace("-PERP", "").replace("USDT", "USD").replace("/", "").strip()
        mapped = DEFAULT_SYMBOL_MAP.get(raw_symbol.upper(), clean)

        # 1. Try direct mapped symbol
        info = mt5.symbol_info(mapped)
        if info:
            if not info.visible:
                mt5.symbol_select(mapped, True)
            return mapped

        # 2. Try common broker prefixes/suffixes (e.g. BTCUSDm, BTCUSD.a, BTCUSD_i)
        all_symbols = [s.name for s in mt5.symbols_get() or []]
        for candidate in all_symbols:
            if clean in candidate.upper():
                if not mt5.symbol_select(candidate, True):
                    continue
                return candidate

        log.warning("Symbol %s (mapped from %s) not found in MT5 symbols list.", mapped, raw_symbol)
        return None

    def calculate_lot(self, symbol: str, balance: float, order_type: int = mt5.ORDER_TYPE_BUY, price: float = 0.0) -> float:
        """Calculate order volume for exactly $100 trade amount (or custom fixed lot/risk)."""
        info = mt5.symbol_info(symbol)
        if not info:
            return 0.01

        min_lot = info.volume_min or 0.01
        max_lot = info.volume_max or 10.0
        step = info.volume_step or 0.01

        # 1. If explicit fixed lot was provided
        if self.lot_size is not None:
            return max(min_lot, min(self.lot_size, max_lot))

        # 2. Exact $100 sizing
        if self.dollar_amount > 0 and price > 0:
            margin_1lot = mt5.order_calc_margin(order_type, symbol, 1.0, price)
            if margin_1lot and margin_1lot > 0:
                raw_lots = self.dollar_amount / margin_1lot
                steps = round(raw_lots / step)
                return max(min_lot, min(round(steps * step, 2), max_lot))

        # 3. Fallback calculation ($100 risk capital per 1000)
        raw_lots = (self.dollar_amount / 1000.0) * 0.1
        steps = round(raw_lots / step)
        return max(min_lot, min(round(steps * step, 2), max_lot))

    def copy_open_position(self, pos: InvoPosition) -> bool:
        """Open a matching position on MT5 when trader enters (guarded by Risk Shield)."""
        # Synchronize closed deals
        self.risk.sync_with_mt5_history()

        # Check daily risk rules
        allowed, reason = self.risk.allowed(self.dollar_amount)
        if not allowed:
            log.warning("🚨 [RISK SHIELD] Copy order blocked by Risk Manager: %s", reason)
            return False

        mt5_symbol = self.resolve_mt5_symbol(pos.symbol)
        if not mt5_symbol:
            log.warning("Cannot copy %s: symbol unavailable on your broker.", pos.symbol)
            return False

        # Check symbol cooldown
        is_locked, remaining_mins = self.risk.is_in_cooldown(mt5_symbol)
        if is_locked:
            log.warning("⏳ [COOLDOWN] Copy trade blocked: %s is in cooldown (%.1fm remaining)", mt5_symbol, remaining_mins)
            return False

        account = mt5.account_info()
        balance = account.balance if account else 10000.0
        lots = self.calculate_lot(mt5_symbol, balance)


        info = mt5.symbol_info(mt5_symbol)
        tick = mt5.symbol_info_tick(mt5_symbol)
        if not info or not tick:
            log.warning("Cannot get price tick for %s", mt5_symbol)
            return False

        digits = info.digits
        point = info.point

        if pos.side == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
            sl = round(pos.stop_loss, digits) if pos.stop_loss else 0.0
            tp = round(pos.take_profit, digits) if pos.take_profit else 0.0
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
            sl = round(pos.stop_loss, digits) if pos.stop_loss else 0.0
            tp = round(pos.take_profit, digits) if pos.take_profit else 0.0

        account = mt5.account_info()
        balance = account.balance if account else 10000.0
        lots = self.calculate_lot(mt5_symbol, balance, order_type, price)

        if self.dry_run:
            simulated_ticket = int(time.time())
            log.info(
                "🧪 [DRY-RUN COPY] Trader: @%s | %s %s | Lots: %.2f ($%.2f) | Price: %.*f | SL: %s | TP: %s | ID: %s",
                pos.trader, pos.side, mt5_symbol, lots, self.dollar_amount, digits, price, sl or 'None', tp or 'None', pos.trade_id
            )
            self.active_copies[pos.trade_id] = {
                "ticket": simulated_ticket,
                "symbol": mt5_symbol,
                "side": pos.side,
                "lots": lots,
                "trader": pos.trader
            }
            self.log_trade_action("OPEN_DRY_RUN", pos.trader, mt5_symbol, pos.side, lots, price, sl, tp, simulated_ticket)
            return True

        # Send live MT5 market order
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": mt5_symbol,
            "volume": lots,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": SETTINGS.slippage_points,
            "magic": self.magic_number,
            "comment": f"Invo:{pos.trader[:10]}",
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
                "🚀 [COPIED ORDER FILLED] Trader: @%s | %s %s | Lots: %.2f ($%.2f) | Price: %.*f | Ticket: %d",
                pos.trader, pos.side, mt5_symbol, lots, self.dollar_amount, digits, result.price, result.order
            )
            self.active_copies[pos.trade_id] = {
                "ticket": result.order,
                "symbol": mt5_symbol,
                "side": pos.side,
                "lots": lots,
                "trader": pos.trader
            }
            self.log_trade_action("OPEN_LIVE", pos.trader, mt5_symbol, pos.side, lots, result.price, sl, tp, result.order)
            return True
        else:
            err = result.comment if result else mt5.last_error()
            log.warning("❌ Failed to copy order for %s (%s): %s", mt5_symbol, pos.side, err)
            return False

    def close_copied_position(self, trade_id: str):
        """Close MT5 position when trader closes on Invo."""
        copy_info = self.active_copies.get(trade_id)
        if not copy_info:
            return

        ticket = copy_info["ticket"]
        symbol = copy_info["symbol"]
        side = copy_info["side"]
        lots = copy_info["lots"]
        trader = copy_info["trader"]

        log.info(" Trader @%s closed position %s. Closing MT5 position (Ticket: %s)...", trader, trade_id, ticket)

        if self.dry_run:
            log.info(" [DRY-RUN CLOSE] Closed ticket %s for %s", ticket, symbol)
            self.log_trade_action("CLOSE_DRY_RUN", trader, symbol, side, lots, 0.0, 0.0, 0.0, ticket)
            del self.active_copies[trade_id]
            return

        # Find open position in MT5
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            log.info("Position %s is already closed on MT5.", ticket)
            del self.active_copies[trade_id]
            return

        pos = positions[0]
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(symbol)
        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": symbol,
            "volume": pos.volume,
            "type": close_type,
            "price": price,
            "deviation": SETTINGS.slippage_points,
            "magic": self.magic_number,
            "comment": "Invo:Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info(" [POSITION CLOSED] Ticket %s on %s closed cleanly.", ticket, symbol)
            self.log_trade_action("CLOSE_LIVE", trader, symbol, side, pos.volume, result.price, 0.0, 0.0, ticket)
            del self.active_copies[trade_id]
        else:
            err = result.comment if result else mt5.last_error()
            log.warning("Failed to close position %s: %s", ticket, err)

    def log_trade_action(self, action, trader, symbol, side, lots, price, sl, tp, ticket):
        path = Path(SETTINGS.log_file)
        new_file = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["timestamp", "action", "trader", "symbol", "side", "lots", "price", "sl", "tp", "ticket"])
            w.writerow([datetime.now().isoformat(), action, trader, symbol, side, lots, price, sl, tp, ticket])

    def run(self):
        """Main continuous polling and copying loop."""
        if not self.target_traders:
            log.error("No target traders configured. Specify at least one trader to copy (e.g. --trader username).")
            return

        self.connect_mt5()
        log.info(
            " Invo Copy Trading Active | Tracking %d Traders: %s | Amount: $%.2f | Mode: %s",
            len(self.target_traders), ", ".join(self.target_traders), self.dollar_amount,
            "DRY-RUN (Simulated)" if self.dry_run else "LIVE MT5 TRADING"
        )

        try:
            while True:
                current_poll_trade_ids = set()

                for trader in self.target_traders:
                    try:
                        open_positions = self.client.fetch_trader_positions(trader)
                        for pos in open_positions:
                            current_poll_trade_ids.add(pos.trade_id)

                            # New position opened by trader
                            if pos.trade_id not in self.known_trader_positions:
                                log.info("🎯 NEW SIGNAL DETECTED from @%s: %s %s @ %.4f", pos.trader, pos.side, pos.symbol, pos.entry_price)
                                success = self.copy_open_position(pos)
                                if success:
                                    self.known_trader_positions.add(pos.trade_id)

                    except Exception as exc:
                        log.warning("Error fetching positions for trader @%s: %s", trader, exc)

                # Check if any previously copied positions were closed by the trader
                if self.close_on_exit:
                    closed_ids = [tid for tid in list(self.active_copies.keys()) if tid not in current_poll_trade_ids]
                    for cid in closed_ids:
                        self.close_copied_position(cid)
                        if cid in self.known_trader_positions:
                            self.known_trader_positions.remove(cid)

                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            log.info("Invo Copier stopped by user.")
        finally:
            mt5.shutdown()
            log.info("MT5 connection closed.")


def main():
    parser = argparse.ArgumentParser(description="Invo App to MetaTrader 5 Autonomous Copy Trader")
    parser.add_argument("--trader", nargs="+", default=list(SETTINGS.invo_target_traders) or ["top_trader"], help="One or more Invo trader usernames / profile IDs to copy")
    parser.add_argument("--amount", type=float, default=100.0, help="Dollar amount to trade per position (default: $100.00)")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Polling interval in seconds (default: 1.0s)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate copying without sending live broker orders")
    parser.add_argument("--lot-size", type=float, default=None, help="Fixed lot size (overrides dollar amount if set)")
    parser.add_argument("--no-close-sync", action="store_true", help="Do not close MT5 position when trader closes on Invo")

    args = parser.parse_args()

    copier = InvoMT5Copier(
        target_traders=args.trader,
        poll_interval=args.poll_interval,
        dry_run=args.dry_run,
        dollar_amount=args.amount,
        lot_size=args.lot_size,
        close_on_exit=not args.no_close_sync
    )
    copier.run()


if __name__ == "__main__":
    main()

