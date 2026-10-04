import numpy as np
import pandas as pd


def overlapping_returns(pnl, horizon=10):
    return pnl.rolling(horizon).sum().dropna()


def initial_margin(pnl, notional, level=0.99, horizon=10, window=500,
                   stress_weight=0.25, floor_pct=0.0):
    r = overlapping_returns(pnl, horizon)
    recent = r.iloc[-window:]
    base = -np.quantile(recent, 1 - level)
    stressed = -np.quantile(r, 1 - level)
    blended = (1 - stress_weight) * base + stress_weight * stressed
    return max(blended, floor_pct) * notional


def margin_series(pnl, notional, level=0.99, horizon=10, window=500,
                  stress_weight=0.25, step=5):
    r = overlapping_returns(pnl, horizon)
    rows = []
    for i in range(window, len(r), step):
        sub = r.iloc[:i]
        recent = sub.iloc[-window:]
        base = -np.quantile(recent, 1 - level)
        stressed = -np.quantile(sub, 1 - level)
        im = ((1 - stress_weight) * base + stress_weight * stressed) * notional
        rows.append((r.index[i - 1], im))
    return pd.Series(dict(rows), name="initial_margin")


def margin_call_report(im_series, tolerance=0.10):
    change = im_series.pct_change()
    flags = change.abs() > tolerance
    return pd.DataFrame({
        "initial_margin": im_series.round(0),
        "pct_change": (change * 100).round(2),
        "call_flag": flags,
    })


def margin_coverage(pnl, im_series, notional, horizon=10):
    realised = overlapping_returns(pnl, horizon) * notional
    aligned = im_series.reindex(realised.index, method="ffill").shift(1).dropna()
    realised = realised.reindex(aligned.index)
    breaches = (realised < -aligned).sum()
    return int(breaches), len(aligned), breaches / len(aligned)
