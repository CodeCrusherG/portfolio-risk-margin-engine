# Portfolio Risk and Margin Engine

Python tool that computes portfolio risk, back-tests it, calculates initial margin and writes an Excel report.

## Features
- SQL data layer (SQLite) for historical prices; yfinance loader with synthetic fallback
- 1-day 99% VaR and Expected Shortfall: historical, parametric, Monte Carlo
- Back-testing: rolling historical VaR, Kupiec POF test, Basel traffic-light zones
- VaR-based initial margin (99%, 10-day MPOR via sqrt-time scaling) and auto-generated daily commentary
- Scenario stress tests; Excel report and back-test plot

## Run
    pip install numpy pandas scipy matplotlib openpyxl yfinance
    python run_report.py

Set `TICKERS` in `run_report.py` to use real market data.

## Files
- `risk_engine.py` - models, back-testing, margin, commentary
- `run_report.py` - pipeline, Excel report (`risk_report.xlsx`), plot (`backtest_plot.png`)
