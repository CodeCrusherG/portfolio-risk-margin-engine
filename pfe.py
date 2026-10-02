"""Potential Future Exposure (PFE) for an interest rate swap under Vasicek.

Simulates short-rate paths, revalues a payer swap along each path using
closed-form Vasicek zero-coupon bonds, and reports Expected Exposure (EE),
PFE at a chosen percentile, and EPE.

Simplifications (state these in interviews):
- single trade, no netting, no collateral (unmargined exposure)
- annual resets/payments; floating coupon fixed at the Vasicek 1y simple rate
  implied by the short rate on the last reset date
- Vasicek parameters are illustrative, not calibrated
"""
import numpy as np


def vasicek_bond(r, tau, a, b, sigma):
    """Zero-coupon bond price P(t, t+tau) given short rate r."""
    tau = np.asarray(tau, dtype=float)
    B = (1 - np.exp(-a * tau)) / a
    A = np.exp((b - sigma**2 / (2 * a**2)) * (B - tau) - sigma**2 * B**2 / (4 * a))
    return A * np.exp(-B * r)


def simulate_short_rate(r0, a, b, sigma, T, steps_per_year, n_paths, seed=1):
    """Exact Vasicek discretisation. Returns (times, rates[n_paths, n_steps+1])."""
    rng = np.random.default_rng(seed)
    dt = 1 / steps_per_year
    n = int(T * steps_per_year)
    times = np.arange(n + 1) * dt
    r = np.empty((n_paths, n + 1))
    r[:, 0] = r0
    decay = np.exp(-a * dt)
    sd = sigma * np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a))
    for i in range(n):
        r[:, i + 1] = b + (r[:, i] - b) * decay + sd * rng.standard_normal(n_paths)
    return times, r


def par_swap_rate(r0, a, b, sigma, maturity):
    pay = np.arange(1, maturity + 1)
    P = vasicek_bond(r0, pay, a, b, sigma)
    return (1 - P[-1]) / P.sum()


def swap_value(r, r_reset, t, K, maturity, a, b, sigma, notional):
    """Payer swap (pay fixed K, receive floating) value at time t per path.

    r:       short rate at time t (per path)
    r_reset: short rate at the last reset date (fixes the current coupon)
    """
    pay = np.arange(1, maturity + 1)
    remaining = pay[pay > t + 1e-9]
    if len(remaining) == 0:
        return np.zeros_like(r)
    s = np.floor(t + 1e-9)                      # last reset date
    coupon = 1.0 / vasicek_bond(r_reset, 1.0, a, b, sigma) - 1.0
    P_next = vasicek_bond(r, s + 1 - t, a, b, sigma)
    P = vasicek_bond(r[:, None], (remaining - t)[None, :], a, b, sigma)
    floating = (1.0 + coupon) * P_next - P[:, -1]
    return notional * (floating - K * P.sum(axis=1))


def exposure_profile(notional=10_000_000, maturity=5, r0=0.065, a=0.15,
                     b=0.065, sigma=0.012, n_paths=10_000, pct=0.95,
                     steps_per_year=12, seed=1):
    times, rates = simulate_short_rate(r0, a, b, sigma, maturity,
                                       steps_per_year, n_paths, seed)
    K = par_swap_rate(r0, a, b, sigma, maturity)
    cols = []
    for i, t in enumerate(times):
        reset_idx = int(round(np.floor(t + 1e-9) * steps_per_year))
        cols.append(swap_value(rates[:, i], rates[:, reset_idx], t, K,
                               maturity, a, b, sigma, notional))
    mtm = np.column_stack(cols)
    expo = np.maximum(mtm, 0.0)
    return {
        "times": times,
        "fixed_rate": K,
        "mtm": mtm,
        "EE": expo.mean(axis=0),
        "PFE": np.quantile(expo, pct, axis=0),
        "EPE": float(expo.mean(axis=0).mean()),
        "peak_PFE": float(np.quantile(expo, pct, axis=0).max()),
        "peak_PFE_time": float(times[np.quantile(expo, pct, axis=0).argmax()]),
        "pct": pct,
    }


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = exposure_profile()
    print(f"Par fixed rate: {res['fixed_rate']*100:.3f}%")
    print(f"EPE: {res['EPE']:,.0f}")
    print(f"Peak {res['pct']*100:.0f}% PFE: {res['peak_PFE']:,.0f} "
          f"at t = {res['peak_PFE_time']:.2f}y")
    plt.figure(figsize=(9, 5))
    plt.plot(res["times"], res["PFE"], label=f"PFE {res['pct']*100:.0f}%")
    plt.plot(res["times"], res["EE"], label="Expected Exposure")
    plt.xlabel("Years")
    plt.ylabel("Exposure")
    plt.title("5y payer IRS exposure profile (Vasicek)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pfe_profile.png", dpi=130)
