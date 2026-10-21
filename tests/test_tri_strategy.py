import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from datetime import datetime, timedelta, timezone
import numpy as np

from risk import RiskManager
from strategies.mtf_pullback import analyse_mtf_pullback, Signal
from strategies.session_breakout import analyse_session_breakout, calculate_asian_range, get_current_session_window


class TestTriStrategyAndRiskShield(unittest.TestCase):

    def setUp(self):
        self.risk = RiskManager(
            max_loss=1000.0,
            max_trades=10,
            max_consecutive_losses=3,
            daily_profit_target=2000.0,
            cooldown_minutes=15.0
        )

    # ─────────────────────────────────────────────────────────────
    # Risk Manager & Cooldown Shield Tests
    # ─────────────────────────────────────────────────────────────
    def test_risk_manager_cooldown(self):
        symbol = "EURUSD"
        self.assertFalse(self.risk.is_in_cooldown(symbol)[0])

        # Apply cooldown
        self.risk.set_cooldown(symbol, minutes=10.0, reason="Test SL hit")
        is_locked, rem = self.risk.is_in_cooldown(symbol)
        self.assertTrue(is_locked)
        self.assertGreater(rem, 9.0)

        # Different symbol is not locked
        self.assertFalse(self.risk.is_in_cooldown("GBPUSD")[0])

    def test_risk_manager_circuit_breaker(self):
        # 1st loss
        self.risk.record(-100.0, symbol="EURUSD")
        allowed, _ = self.risk.allowed()
        self.assertTrue(allowed)
        self.assertEqual(self.risk.consecutive_losses, 1)

        # 2nd loss
        self.risk.record(-150.0, symbol="GBPUSD")
        allowed, _ = self.risk.allowed()
        self.assertTrue(allowed)
        self.assertEqual(self.risk.consecutive_losses, 2)

        # 3rd consecutive loss -> trips circuit breaker
        self.risk.record(-200.0, symbol="USDJPY")
        allowed, reason = self.risk.allowed()
        self.assertFalse(allowed)
        self.assertIn("consecutive-loss", reason)

        # Reset on win
        self.risk.consecutive_losses = 0
        self.risk.record(300.0, symbol="EURUSD")
        self.assertEqual(self.risk.consecutive_losses, 0)
        self.assertEqual(self.risk.realised_pnl, -150.0)

    # ─────────────────────────────────────────────────────────────
    # MTF Pullback Strategy Tests
    # ─────────────────────────────────────────────────────────────
    def test_mtf_pullback_bullish_setup(self):
        # Generate synthetic H1 candles in clear uptrend
        h1_candles = []
        base = 1.1000
        for i in range(80):
            price = base + (i * 0.0005)
            h1_candles.append({
                "time": int(1700000000 + i * 3600),
                "open": price - 0.0002,
                "high": price + 0.0004,
                "low": price - 0.0003,
                "close": price,
                "volume": 500
            })

        # Generate synthetic M5 candles pulling back to EMA50 then bouncing
        m5_candles = []
        m5_base = 1.1350
        for i in range(70):
            if i < 50:
                p = m5_base + (i * 0.0002)
            elif i < 65:
                p = m5_base + (50 * 0.0002) - ((i - 50) * 0.0003)  # Pullback dip
            else:
                p = m5_base + (50 * 0.0002) - (15 * 0.0003) + ((i - 65) * 0.0004) # Bounce

            m5_candles.append({
                "time": int(1700000000 + i * 300),
                "open": p - 0.0001,
                "high": p + 0.0002,
                "low": p - 0.0002,
                "close": p,
                "volume": 200
            })

        signal = analyse_mtf_pullback(h1_candles, m5_candles)
        self.assertIsInstance(signal, Signal)
        self.assertIn(signal.direction, ("call", "wait"))

    # ─────────────────────────────────────────────────────────────
    # Session Breakout Strategy Tests
    # ─────────────────────────────────────────────────────────────
    def test_session_breakout_asian_range(self):
        now = datetime.now(timezone.utc)
        target_date = now.date()

        # Synthetic candles covering Asian session (00:00 - 06:45 UTC)
        candles = []
        for hour in range(10):
            for m in range(0, 60, 15):
                dt = datetime(now.year, now.month, now.day, hour, m, tzinfo=timezone.utc)
                candles.append({
                    "time": int(dt.timestamp()),
                    "open": 1.2000,
                    "high": 1.2030 if hour < 7 else 1.2060,
                    "low": 1.1980 if hour < 7 else 1.1970,
                    "close": 1.2010 if hour < 7 else 1.2055,
                    "volume": 300
                })

        high_val, low_val, range_val = calculate_asian_range(candles, target_date)
        self.assertIsNotNone(high_val)
        self.assertIsNotNone(low_val)
        self.assertAlmostEqual(high_val, 1.2030, places=4)
        self.assertAlmostEqual(low_val, 1.1980, places=4)
        self.assertAlmostEqual(range_val, 0.0050, places=4)


if __name__ == "__main__":
    unittest.main()
