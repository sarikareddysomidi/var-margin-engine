import numpy as np
from scipy.stats import norm


def historical_var(pnl, level=0.99):
    return -np.quantile(pnl, 1 - level)


def expected_shortfall(pnl, level=0.99):
    cutoff = np.quantile(pnl, 1 - level)
    tail = pnl[pnl <= cutoff]
    return -tail.mean()


def parametric_var(pnl, level=0.99):
    return -(pnl.mean() + norm.ppf(1 - level) * pnl.std(ddof=1))
