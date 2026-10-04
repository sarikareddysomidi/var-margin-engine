import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import simulate_returns, portfolio_returns
from src.var import historical_var, expected_shortfall, parametric_var
from src.backtest import rolling_var, summarize
from src.margin import margin_series, margin_call_report, margin_coverage

NOTIONAL = 100_000_000
WEIGHTS = {"EQ_INDEX": 0.40, "GOVT_BOND": 0.30, "FX_USDINR": 0.15, "CREDIT_ETF": 0.15}


def main():
    os.makedirs("output", exist_ok=True)
    returns = simulate_returns()
    pnl = portfolio_returns(returns, WEIGHTS)

    print("1-day 99% risk measures on full sample")
    print("  historical VaR :", round(historical_var(pnl.values) * NOTIONAL))
    print("  parametric VaR :", round(parametric_var(pnl.values) * NOTIONAL))
    print("  expected shortfall:", round(expected_shortfall(pnl.values) * NOTIONAL))

    results = {}
    for method in ("historical", "parametric"):
        v = rolling_var(pnl, method=method)
        results[method] = v
        print(f"\nBack-test ({method})")
        for k, val in summarize(pnl, v).items():
            print(f"  {k}: {val}")

    im = margin_series(pnl, NOTIONAL)
    report = margin_call_report(im)
    report.to_csv("output/margin_report.csv")
    b, n, rate = margin_coverage(pnl, im, NOTIONAL)
    print(f"\n10-day margin breaches: {b}/{n} ({rate:.2%})")
    print("margin calls flagged:", int(report["call_flag"].sum()))

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    ax[0].plot(pnl.index, pnl * NOTIONAL, lw=0.5, label="daily P&L")
    ax[0].plot(results["historical"].index, -results["historical"] * NOTIONAL, label="-VaR (hist)")
    ax[0].plot(results["parametric"].index, -results["parametric"] * NOTIONAL, label="-VaR (param)")
    ax[0].legend()
    ax[0].set_title("P&L vs 99% VaR")
    ax[1].plot(im.index, im, label="initial margin")
    ax[1].set_title("Initial margin (10-day MPOR, 99%)")
    plt.tight_layout()
    plt.savefig("output/var_margin.png", dpi=130)


if __name__ == "__main__":
    main()
