# Markowitz Portfolio Optimization

A Python implementation of the **Markowitz Mean-Variance Model** for portfolio optimization, applied to a basket of large-cap NSE (Indian) stocks. The project simulates thousands of random portfolios, visualizes the risk-return tradeoff (efficient frontier), and uses numerical optimization to find the portfolio with the maximum Sharpe ratio.

## Overview

Modern Portfolio Theory (Markowitz, 1952) states that for a given level of risk, there exists an optimal combination of assets that maximizes expected return. This project builds that theory from scratch — no pre-built portfolio optimization libraries — using only `numpy`, `pandas`, and `scipy`.

The model:
1. Pulls historical price data for 18 NSE-listed stocks
2. Computes daily log returns and annualizes them
3. Generates 10,000 random portfolios (Monte Carlo simulation) to visualize the risk-return space
4. Uses `scipy.optimize` (SLSQP) to solve for the portfolio weights that maximize the Sharpe ratio
5. Plots the efficient frontier with the optimal portfolio highlighted

## Stocks Used

18 large-cap NSE stocks spanning banking, FMCG, IT, auto, pharma, and infrastructure:

`COLPAL`, `ICICIBANK`, `LICHSGFIN`, `LT`, `IRFC`, `RELIANCE`, `TCS`, `HDFCBANK`, `HINDUNILVR`, `INFY`, `ITC`, `KOTAKBANK`, `MARUTI`, `NESTLEIND`, `SBIN`, `SUNPHARMA`, `TITAN`, `ULTRACEMCO`

Historical data range: **2012–2026** (daily close prices via Yahoo Finance)

## Tech Stack

- `numpy` — matrix operations, portfolio math
- `pandas` — data handling and returns calculation
- `yfinance` — historical stock price data
- `matplotlib` — visualization (price history, efficient frontier)
- `scipy.optimize` — constrained optimization (SLSQP)

## Methodology

**1. Log Returns**
Daily returns are computed as logarithmic returns rather than simple returns, since log returns are time-additive and better approximate normally distributed returns over short intervals.

**2. Random Portfolio Generation**
10,000 portfolios are generated with random weights (constrained to sum to 1) to map out the achievable risk-return combinations — this forms the visual "cloud" from which the efficient frontier emerges.

**3. Portfolio Statistics**
For each portfolio:
- **Expected Return** = weighted sum of annualized mean returns
- **Volatility (Risk)** = portfolio standard deviation, computed via the weight vector and the annualized covariance matrix
- **Sharpe Ratio** = return / volatility (risk-free rate assumed at 0 for simplicity)

**4. Optimization**
The optimal portfolio is found by minimizing the *negative* Sharpe ratio (since `scipy.optimize.minimize` only minimizes) subject to:
- Weights sum to 1 (`eq` constraint)
- No short-selling — each weight bounded between 0 and 1

## Installation

```bash
pip install numpy pandas yfinance matplotlib scipy
```

## Usage

```bash
python Markowitz_1.py
```

Running the script will:
1. Download historical price data for all 18 stocks
2. Plot the raw price history
3. Compute and print log returns
4. Simulate 10,000 random portfolios and plot them (Volatility vs. Return, colored by Sharpe ratio)
5. Run the optimizer and print the optimal portfolio weights, expected return, volatility, and Sharpe ratio
6. Re-plot the efficient frontier with the optimal portfolio marked as a green star

## Sample Output

```
Optimal portfolio:  [0.12  0.00  0.08  0.15  0.00  ...]
Expected return, volatility and Sharpe ratio: [0.184  0.162  1.135]
```

*(exact values will vary based on the data pulled at runtime)*

## Project Structure

```
├── Markowitz_1.py      # Main script — data pipeline, simulation, optimization, plotting
└── README.md
```

## Future Improvements

- [ ] Add a risk-free rate input for a more accurate Sharpe ratio calculation
- [ ] Include short-selling as an optional constraint
- [ ] Extend to a rolling-window backtest to test stability of optimal weights over time
- [ ] Compare Monte Carlo results against the analytical efficient frontier solution
- [ ] Add Sortino ratio and max drawdown as additional risk metrics

## Disclaimer

This project is for educational purposes only and does not constitute financial advice. Historical performance does not guarantee future results.

## Author

Built by Hriday as part of self-directed quant finance preparation.
