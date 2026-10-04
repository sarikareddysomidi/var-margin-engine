import numpy as np
import pandas as pd

ASSETS = ["EQ_INDEX", "GOVT_BOND", "FX_USDINR", "CREDIT_ETF"]


def simulate_returns(n_days=1500, seed=7):
    rng = np.random.default_rng(seed)
    vol = np.array([0.012, 0.004, 0.003, 0.006])
    corr = np.array([
        [1.00, -0.25, 0.20, 0.55],
        [-0.25, 1.00, -0.10, -0.15],
        [0.20, -0.10, 1.00, 0.10],
        [0.55, -0.15, 0.10, 1.00],
    ])
    chol = np.linalg.cholesky(corr)
    dof = 5
    z = rng.standard_t(dof, size=(n_days, len(vol))) / np.sqrt(dof / (dof - 2))
    scale = np.ones(n_days)
    scale[900:1000] = 2.2
    r = (z @ chol.T) * vol * scale[:, None]
    idx = pd.bdate_range("2019-01-01", periods=n_days)
    return pd.DataFrame(r, index=idx, columns=ASSETS)


def portfolio_returns(returns, weights):
    w = pd.Series(weights).reindex(returns.columns).values
    return pd.Series(returns.values @ w, index=returns.index, name="portfolio")
