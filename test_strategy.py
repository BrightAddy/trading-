import unittest
import numpy as np
from strategy import ema, rsi, rsi_single, atr, macd, bollinger_bands, adx, stochastic, analyse, Signal
from risk import RiskManager

class TestTradingEngine(unittest.TestCase):

    def test_indicators(self):
        prices = np.linspace(100, 110, 50)
        e = ema(prices, 9)
        self.assertEqual(len(e), 50)
        self.assertGreater(e[-1], e[0])

        r = rsi(prices, 14)
        self.assertEqual(len(r), 50)
        self.assertGreater(r[-1], 50.0)

        r_single = rsi_single(prices, 14)
        self.assertGreater(r_single, 50.0)

        highs = prices + 1.0
        lows = prices - 1.0
        a = atr(highs, lows, prices, 14)
        self.assertGreater(a, 0.0)

        macd_l, macd_s, macd_h = macd(prices)
        self.assertEqual(len(macd_l), 50)

        upper, mid, lower = bollinger_bands(prices, 20)
        self.assertEqual(len(upper), 50)
        self.assertGreater(upper[-1], lower[-1])

    def test_stochastic(self):
        np.random.seed(42)
        h = np.cumsum(np.random.randn(50)) + 100
        l = h - np.abs(np.random.randn(50))
        c = (h + l) / 2
        k, d = stochastic(h, l, c)
        self.assertGreaterEqual(k, 0.0)
        self.assertLessEqual(k, 100.0)
        self.assertGreaterEqual(d, 0.0)
        self.assertLessEqual(d, 100.0)

    def test_risk_manager(self):
        rm = RiskManager(max_loss=500, max_trades=5, max_consecutive_losses=2, daily_profit_target=1000)
        
        # Test initial state
        allowed, reason = rm.allowed(100)
        self.assertTrue(allowed)
        self.assertEqual(reason, "ok")

        # Record consecutive losses
        rm.record(-100)
        rm.record(-100)
        allowed, reason = rm.allowed(100)
        self.assertFalse(allowed)
        self.assertIn("consecutive-loss", reason)

    def test_strategy_insufficient_history(self):
        candles = [{"time": i, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05} for i in range(50)]
        sig = analyse(candles)
        self.assertEqual(sig.direction, "wait")
        self.assertIn("insufficient_history", sig.reasons)

    def test_strategy_returns_signal_dataclass(self):
        """Strategy analyse always returns a Signal object."""
        candles = [{"time": i, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05} for i in range(250)]
        sig = analyse(candles)
        self.assertIsInstance(sig, Signal)
        self.assertIn(sig.direction, ("call", "put", "wait"))

if __name__ == "__main__":
    unittest.main()
