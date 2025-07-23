# Macro-Sensitivity-Bond-Funds

This repository contains the full implementation of the empirical analysis conducted in the Master's thesis:

> **Macro-Factor Sensitivity of U.S. Fixed Income Funds: An Empirical Analysis**  
> _Saba Ramezani, ESADE Master in Finance, 2025_

## 📘 Project Summary

This project studies how U.S. fixed-income mutual funds and ETFs respond to macroeconomic variables such as:

- Interest rates (Fed Funds & 10-Year Treasury)
- Inflation (Consumer Price Index)
- Credit risk (BAA-Treasury spreads)
- Economic growth (Industrial Production)
- Labor market conditions (Unemployment Rate)

Using data from 2015–2024, the project applies both **static regressions** and **rolling-window regressions** to measure macro-factor sensitivities and translate them into actionable portfolio strategies.

## Key Features

- Fund-level sensitivity estimation using OLS with HAC standard errors
- Rolling 36-month regressions to track time-varying macro exposures
- Construction of three rule-based strategies:
  - **Procyclical Growth Portfolio**
  - **Defensive Portfolio**
  - **Inflation-Hedged Portfolio**
- Portfolio simulation, risk/return evaluation, and visualization

## Methodology

1. **Data Collection**
   - Monthly fund returns via `yfinance`
   - Macroeconomic indicators via `fredapi`

2. **Regression Analysis**
   - Estimate factor loadings for each fund using macro variables
   - Assess explanatory power (Adj. R², t-stats, etc.)

3. **Portfolio Construction**
   - Weight funds based on their estimated macro sensitivities
   - Create targeted portfolios for different macro regimes

4. **Performance Analysis**
   - Compute cumulative returns, volatility, Sharpe ratio, and drawdown
   - Visualize portfolio return paths, risk/return profiles, and betas

## 🛠Dependencies

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn yfinance fredapi statsmodels scipy xlsxwriter
```
Important: You need a FRED API key. Add your key in main.py where indicated.

## 📈 Outputs

- `ols_results.csv`: Fund-level static macro betas and t-stats  
- `rolling_betas.xlsx`: Time-series of beta coefficients  
- `portfolio_returns.xlsx`: Monthly returns of macro-sensitive portfolios  
- `performance_stats.csv`: Risk-adjusted performance metrics  

---

## 📚 Thesis

This project supports the Master's thesis submitted to ESADE Business School:

**📄 Macro-Factor Sensitivity of U.S. Fixed Income Funds: An Empirical Analysis**  
**Author:** Saba Ramezani  
**Advisor:** Prof. Jose Suarez-Lledo

---

## 📜 Citation

If you use this code or methodology, please cite the thesis:

```bibtex
@thesis{ramezani2025macro,
  title={Macro-Factor Sensitivity of U.S. Fixed Income Funds: An Empirical Analysis},
  author={Ramezani, Saba},
  year={2025},
  school={ESADE Business School}
}
