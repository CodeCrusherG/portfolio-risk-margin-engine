"""Unit tests. Run with:  python -m unittest -v"""
import unittest

import numpy as np
import pandas as pd

import pfe
import risk_engine as re_


class TestVaRES(unittest.TestCase):
    def setUp(self):
        self.r = np.arange(-50, 50) / 1000.0  # -5.0% ... +4.9%

    def test_hist_var_known_quantile(self):
        self.assertAlmostEqual(re_.hist_var(self.r, 0.99),
                               -np.quantile(self.r, 0.01))

    def test_es_not_less_than_var(self):
        self.assertGreaterEqual(re_.hist_es(self.r, 0.99),
                                re_.hist_var(self.r, 0.99))

    def test_var_increases_with_confidence(self):
        self.assertGreater(re_.hist_var(self.r, 0.99),
                           re_.hist_var(self.r, 0.95))

    def test_parametric_var_matches_normal_formula(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0, 0.01, 200_000)
        self.assertAlmostEqual(re_.param_var(r, 0.99), 0.0233, delta=0.0005)


class TestBacktest(unittest.TestCase):
    def test_kupiec_perfect_calibration_not_rejected(self):
        lr, p = re_.kupiec_pof(10, 1000, 0.99)
        self.assertAlmostEqual(lr, 0.0, places=6)
        self.assertGreater(p, 0.9)

    def test_kupiec_rejects_too_many_exceptions(self):
        _, p = re_.kupiec_pof(40, 1000, 0.99)
        self.assertLess(p, 0.01)

    def test_kupiec_zero_exceptions_finite(self):
        lr, p = re_.kupiec_pof(0, 250, 0.99)
        self.assertTrue(np.isfinite(lr))

    def test_traffic_light_boundaries(self):
        self.assertEqual(re_.traffic_light(4), "GREEN")
        self.assertEqual(re_.traffic_light(5), "YELLOW")
        self.assertEqual(re_.traffic_light(9), "YELLOW")
        self.assertEqual(re_.traffic_light(10), "RED")

    def test_rolling_var_uses_no_lookahead(self):
        rng = np.random.default_rng(1)
        r = pd.Series(rng.normal(0, 0.01, 400))
        var = re_.rolling_hist_var(r, window=250, alpha=0.99)
        first = var.index[0]
        expected = re_.hist_var(r.iloc[first - 250:first].values, 0.99)
        self.assertAlmostEqual(var.iloc[0], expected)


class TestMargin(unittest.TestCase):
    def test_initial_margin_sqrt_time_scaling(self):
        rng = np.random.default_rng(2)
        r = pd.Series(rng.normal(0, 0.01, 600))
        im10 = re_.initial_margin(r, 1_000_000, horizon=10)
        im1 = re_.initial_margin(r, 1_000_000, horizon=1)
        self.assertAlmostEqual(im10 / im1, np.sqrt(10))


class TestPFE(unittest.TestCase):
    def test_bond_price_in_unit_interval(self):
        p = pfe.vasicek_bond(0.06, np.array([1, 5, 10]), 0.15, 0.065, 0.012)
        self.assertTrue(np.all((p > 0) & (p < 1)))
        self.assertTrue(np.all(np.diff(p) < 0))  # longer maturity, lower price

    def test_swap_is_zero_value_at_inception(self):
        res = pfe.exposure_profile(n_paths=2000)
        self.assertAlmostEqual(res["mtm"][0, 0], 0.0, places=4)

    def test_exposure_nonnegative_and_pfe_above_ee(self):
        res = pfe.exposure_profile(n_paths=5000)
        self.assertTrue(np.all(res["EE"] >= 0))
        self.assertTrue(np.all(res["PFE"] >= res["EE"] - 1e-6))

    def test_exposure_zero_at_maturity(self):
        res = pfe.exposure_profile(n_paths=2000)
        self.assertAlmostEqual(res["EE"][-1], 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
