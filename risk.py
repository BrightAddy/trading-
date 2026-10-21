import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, Optional, Set, Tuple
import MetaTrader5 as mt5

log = logging.getLogger("risk-manager")

@dataclass
class RiskManager:
    max_loss: float
    max_trades: int
    max_consecutive_losses: int
    daily_profit_target: float = 20000.0
    day: date = field(default_factory=date.today)
    realised_pnl: float = 0.0
    trades: int = 0
    consecutive_losses: int = 0
    peak_pnl: float = 0.0
    max_drawdown: float = 0.0
    cooldown_minutes: float = 30.0
    
    # Internal tracking
    _cooldowns: Dict[str, datetime] = field(default_factory=dict)
    _processed_deal_tickets: Set[int] = field(default_factory=set)

    def refresh_day(self):
        today = date.today()
        if self.day != today:
            log.info("🗓️ New trading day started (%s). Resetting daily P&L and risk counters.", today)
            self.day = today
            self.realised_pnl = 0.0
            self.trades = 0
            self.consecutive_losses = 0
            self.peak_pnl = 0.0
            self.max_drawdown = 0.0
            self._processed_deal_tickets.clear()
            self._cooldowns.clear()

    def set_cooldown(self, symbol: str, minutes: Optional[float] = None, reason: str = ""):
        """Lock out a symbol from new entries for N minutes."""
        dur = minutes if minutes is not None else self.cooldown_minutes
        unlock_time = datetime.now() + timedelta(minutes=dur)
        clean_sym = symbol.upper().strip()
        self._cooldowns[clean_sym] = unlock_time
        log.info(
            "⏳ [COOLDOWN APPLIED] %s locked for %.1f min (until %s) | Reason: %s",
            clean_sym, dur, unlock_time.strftime("%H:%M:%S"), reason or "Trade closed"
        )

    def is_in_cooldown(self, symbol: str) -> Tuple[bool, float]:
        """Check if symbol is currently in cooldown. Returns (is_locked, remaining_minutes)."""
        clean_sym = symbol.upper().strip()
        if clean_sym in self._cooldowns:
            unlock_time = self._cooldowns[clean_sym]
            remaining = (unlock_time - datetime.now()).total_seconds() / 60.0
            if remaining > 0:
                return True, round(remaining, 1)
            else:
                del self._cooldowns[clean_sym]
        return False, 0.0

    def allowed(self, next_amount: float = 0.0) -> Tuple[bool, str]:
        """Verify if new trading is permitted under daily risk limits."""
        self.refresh_day()
        if self.trades >= self.max_trades:
            return False, f"daily trade limit reached ({self.trades}/{self.max_trades})"
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, f"consecutive-loss circuit breaker tripped ({self.consecutive_losses}/{self.max_consecutive_losses})"
        if self.realised_pnl >= self.daily_profit_target:
            return False, f"daily profit target achieved (+${self.realised_pnl:.2f})"
        if self.realised_pnl <= -self.max_loss:
            return False, f"daily loss limit reached (-${self.max_loss:.2f})"
        if next_amount > 0 and (self.realised_pnl - next_amount < -self.max_loss):
            return False, f"next trade could exceed daily loss limit (-${self.max_loss:.2f})"
        return True, "ok"

    def record(self, pnl: float, symbol: Optional[str] = None):
        """Record trade result and update risk statistics."""
        self.refresh_day()
        self.trades += 1
        self.realised_pnl += pnl
        if self.realised_pnl > self.peak_pnl:
            self.peak_pnl = self.realised_pnl
        drawdown = self.peak_pnl - self.realised_pnl
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        if pnl < 0:
            self.consecutive_losses += 1
            log.warning(
                "📉 Recorded Loss: $%.2f | Daily P&L: $%.2f | Consecutive Losses: %d/%d",
                pnl, self.realised_pnl, self.consecutive_losses, self.max_consecutive_losses
            )
        else:
            self.consecutive_losses = 0
            log.info(
                "📈 Recorded Win: +$%.2f | Daily P&L: $%.2f | Consecutive Losses Reset",
                pnl, self.realised_pnl
            )

        if symbol:
            self.set_cooldown(symbol, self.cooldown_minutes, f"Trade completed (${pnl:+.2f})")

    def sync_with_mt5_history(self) -> int:
        """
        Poll MT5 closed deals since start of today and update risk metrics for any
        deals not yet accounted for.
        """
        self.refresh_day()
        if not mt5.account_info():
            return 0

        # Query deals since start of current day
        today_start = datetime.combine(self.day, datetime.min.time())
        now = datetime.now() + timedelta(minutes=5)
        deals = mt5.history_deals_get(today_start, now)
        if not deals:
            return 0

        new_deals_count = 0
        for d in deals:
            # entry: 1 is OUT (closing deal), 2 is IN/OUT
            if d.entry in (1, 2) and d.ticket not in self._processed_deal_tickets:
                self._processed_deal_tickets.add(d.ticket)
                profit = float(d.profit) + float(d.swap) + float(d.commission)
                self.record(profit, symbol=d.symbol)
                new_deals_count += 1

        return new_deals_count

    def calculate_stake(self, balance: float, stake_pct: float = 0.02, min_stake: float = 1.0, max_stake: float = 5000.0) -> float:
        """Calculate dynamic stake based on account balance."""
        stake = balance * stake_pct
        return max(min_stake, min(stake, max_stake))
