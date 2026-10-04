import numpy as np
import pandas as pd
from scipy.stats import chi2

from .var import historical_var, parametric_var


def rolling_var(pnl, window=250, level=0.99, method="historical"):
    fn = historical_var if method == "historical" else parametric_var
    out = pd.Series(index=pnl.index, dtype=float)
    for i in range(window, len(pnl)):
        out.iloc[i] = fn(pnl.iloc[i - window:i].values, level)
    return out


def exceptions(pnl, var_series):
    mask = var_series.notna()
    return (pnl[mask] < -var_series[mask]).astype(int)


def kupiec_pof(n_exceptions, n_obs, level=0.99):
    p = 1 - level
    x, n = n_exceptions, n_obs
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    else:
        phat = x / n
        lr = -2 * (
            (n - x) * np.log(1 - p) + x * np.log(p)
            - (n - x) * np.log(1 - phat) - x * np.log(phat)
        )
    return lr, 1 - chi2.cdf(lr, df=1)


def traffic_light(exc_250):
    if exc_250 <= 4:
        return "green"
    if exc_250 <= 9:
        return "yellow"
    return "red"


def summarize(pnl, var_series, level=0.99):
    exc = exceptions(pnl, var_series)
    lr, pval = kupiec_pof(int(exc.sum()), len(exc), level)
    last250 = int(exc.iloc[-250:].sum())
    return {
        "observations": len(exc),
        "exceptions": int(exc.sum()),
        "expected": round(len(exc) * (1 - level), 1),
        "kupiec_lr": round(lr, 3),
        "kupiec_p": round(pval, 4),
        "last_250_exceptions": last250,
        "traffic_light": traffic_light(last250),
    }
