# VaR and Initial Margin Engine

Python implementation of historical and parametric VaR, expected shortfall, regulatory-style back-testing, and a 10-day initial margin model with a stress blend and margin call flags.

## Components

- `src/data.py` correlated fat-tailed return simulator with a stress regime
- `src/var.py` historical VaR, parametric VaR, expected shortfall
- `src/backtest.py` rolling VaR, exception counts, Kupiec POF test, Basel traffic light
- `src/margin.py` overlapping 10-day historical margin, stress blending, margin call report, coverage check
- `run.py` runs everything and writes `output/`

## Run

    pip install -r requirements.txt
    python run.py

## Findings

On the simulated portfolio, historical VaR passes the Kupiec test while parametric (normal) VaR is rejected at 5%, since returns are fat-tailed. Margin coverage is checked against realised 10-day P&L.

## Next steps

- Replace simulated data with real market data
- Add FX and rates risk factors with full revaluation
- Add SIMM-style delta aggregation
