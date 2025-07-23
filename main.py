import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from fredapi import Fred
from scipy.stats import zscore
import statsmodels.api as sm
import os
import numpy as np

pd.set_option('future.no_silent_downcasting', True)


def monthly_returns(file_path, start_date='2015-01-01', end_date='2024-12-31'):
    df_funds = pd.read_csv(file_path)
    tickers = df_funds['Ticker'].dropna().unique().tolist()

    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame()

    monthly_returns = pd.DataFrame()
    for ticker in data.columns:
        monthly_prices = data[ticker].resample('ME').ffill()
        monthly_prices_start = data[ticker].resample('ME').first()
        returns = (monthly_prices - monthly_prices_start) / monthly_prices_start
        monthly_returns[ticker] = returns

    monthly_returns.index = monthly_returns.index.to_period('M').to_timestamp('M')
    return monthly_returns 

def macro_variables(start_date = '2015-01-01', end_date = '2024-12-31'):
    fred = Fred(api_key='1989267303c99734a7bf1e8ab9977494') 

    series_ids = {
        'FedFunds': 'FEDFUNDS',
        '10Y_Treasury': 'GS10',
        'CPI': 'CPIAUCSL',
        'IndustrialProduction': 'INDPRO',
        'UnemploymentRate': 'UNRATE',
        'BAA_Yield': 'BAA10Y'
    }

    start_dt = pd.to_datetime(start_date) - pd.DateOffset(months=1)
    data = {
        name: fred.get_series(series_id, observation_start=start_dt.strftime('%Y-%m-%d'), observation_end=end_date)
        for name, series_id in series_ids.items()
    }

    df = pd.DataFrame(data)

    df['CreditSpread'] = df['BAA_Yield'] - df['10Y_Treasury']

    df = df.resample('ME').last().ffill()

    df_transformed = pd.DataFrame()
    df_transformed['FedFunds'] = df['FedFunds'].diff()
    df_transformed['10Y_Treasury'] = df['10Y_Treasury'].diff()
    df_transformed['UnemploymentRate'] = df['UnemploymentRate'].diff()
    df_transformed['CreditSpread'] = df['CreditSpread'].diff()
    df_transformed['CPI'] = df['CPI'].pct_change() * 100
    df_transformed['IndustrialProduction'] = df['IndustrialProduction'].pct_change() * 100
   
    full_index = pd.date_range(start=start_date, end=end_date, freq='ME')
    df_transformed = df_transformed.reindex(full_index).ffill()

    # df_transformed.to_csv("should_not_na.csv", index=False)

    macro_var_std = df_transformed.apply(zscore)
    
    return macro_var_std


def ols(monthly_returns, macro_var_std):
    results = []
    for fund in monthly_returns.columns:
        df = pd.concat([monthly_returns[fund], macro_var_std], axis=1)
        y = df[fund]
        X = sm.add_constant(df[macro_var_std.columns])

        valid = ~y.isna() & ~X.isna().any(axis=1)
        if valid.sum() < 10:
            continue
        
        model = sm.OLS(y[valid], X[valid]).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
        result = {
            'Fund': fund,
            'Adj_R2': model.rsquared_adj
        }
        for var in model.params.index:
            result[f'Beta_{var}'] = model.params[var]
            result[f'Tstat_{var}'] = model.tvalues[var]

        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv("regressions/ols_results.csv", index=False)
    
    return results_df



def rolling_reg(monthly_returns, macro_var_std, window=36):
    macro_vars = macro_var_std.columns
    rolling_results = {}

    for fund in monthly_returns.columns:
        # print(f"{fund} index range:\n", macro_var_std.index.min(), "to", macro_var_std.index.max())
        # print("Frequency:", pd.infer_freq(macro_var_std.index))

        fund_returns = monthly_returns[[fund]]

        df = pd.concat([fund_returns, macro_var_std], axis=1, join='inner')
        df.columns = [fund] + list(macro_vars) 

        betas = []

        for i in range(len(df) - window + 1):
            window_df = df.iloc[i:i + window]
            y = window_df[fund]
            X = sm.add_constant(window_df[macro_vars])

            if y.isna().any() or X.isna().any().any():
                continue  

            model = sm.OLS(y, X).fit()
            beta = model.params
            beta.name = window_df.index[-1]
            betas.append(beta)

        if betas:
            fund_betas_df = pd.DataFrame(betas)
            rolling_results[fund] = fund_betas_df

    return rolling_results

def save_rolling_results_to_excel(rolling_results, filename='regressions/rolling_betas.xlsx'):
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        for fund, df in rolling_results.items():
            safe_sheet_name = str(fund)[:31].replace('/', '_').replace('\\', '_')
            df.to_excel(writer, sheet_name=safe_sheet_name)


def plot_rolling_betas(rolling_results, output_dir="rolling_beta_plots"):
    os.makedirs(output_dir, exist_ok=True)
    sns.set(style="whitegrid", font_scale=1.2)

    for fund, df in rolling_results.items():
        if df.empty:
            continue

        plt.figure(figsize=(14, 7))

        for i, column in enumerate(df.columns):
            if column.lower() != 'const':
                plt.plot(df.index, df[column], label=column, linewidth=2)

        plt.title(f"Time-Varying Rolling Betas: {fund}", fontsize=16)
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Beta Coefficient", fontsize=14)
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.legend(title="Macro Factors", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fund}_rolling_betas.png", dpi=300)
        plt.close()


def construct_portfolios(rolling_betas, tickers):
    dates = sorted(set().union(*[df.index for df in rolling_betas.values()]))
    portfolios = {
        'InflationHedged': pd.DataFrame(index=dates, columns=tickers).fillna(0),
        'ProcyclicalGrowth': pd.DataFrame(index=dates, columns=tickers).fillna(0),
        'Defensive': pd.DataFrame(index=dates, columns=tickers).fillna(0)
    }

    for date in dates:
        beta_df = pd.DataFrame({fund: df.loc[date] for fund, df in rolling_betas.items() if date in df.index}).T

        # Inflation-Hedged Portfolio
        if 'CPI' in beta_df.columns:
            pos_beta = beta_df[beta_df['CPI'] > 0]
            if not pos_beta.empty:
                weights = pos_beta['CPI'] / pos_beta['CPI'].sum()
                portfolios['InflationHedged'].loc[date, weights.index] = weights

        # Procyclical Growth Portfolio
        if 'IndustrialProduction' in beta_df.columns and 'CreditSpread' in beta_df.columns:
            ip_z = zscore(beta_df['IndustrialProduction'].fillna(0))
            cs_z = zscore(beta_df['CreditSpread'].fillna(0))
            composite = ip_z - cs_z
            composite = pd.Series(composite, index=beta_df.index)
            top_funds = composite[composite > 0]
            if not top_funds.empty:
                weights = top_funds / top_funds.sum()
                portfolios['ProcyclicalGrowth'].loc[date, weights.index] = weights

        # Defensive Portfolio
        if 'FedFunds' in beta_df.columns and 'IndustrialProduction' in beta_df.columns:
            defensive_score = -beta_df['FedFunds'].fillna(0) - beta_df['IndustrialProduction'].fillna(0)
            defensive_score = defensive_score[defensive_score > 0]
            if not defensive_score.empty:
                weights = defensive_score / defensive_score.sum()
                portfolios['Defensive'].loc[date, weights.index] = weights

    return portfolios


def simulate_portfolio_returns(portfolios, monthly_returns):
    portfolio_returns = {}
    for name, weights in portfolios.items():
        rets = []
        for i in range(len(weights.index) - 1):
            date = weights.index[i]
            next_date = weights.index[i+1]
            if next_date not in monthly_returns.index:
                continue
            # use weights from t to return for t+1
            weights_t = weights.loc[date].fillna(0)
            returns_t1 = monthly_returns.loc[next_date].fillna(0)
            ret = np.dot(weights_t.values, returns_t1.values)
            rets.append((next_date, ret))

        portfolio_returns[name] = pd.Series(dict(rets)).sort_index()

    return pd.DataFrame(portfolio_returns)


def save_portfolio_returns_to_excel(portfolio_returns, filename='portfolios/portfolio_returns.xlsx'):
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        for portfolio_name in portfolio_returns.columns:
            df = portfolio_returns[[portfolio_name]].copy()
            df.columns = ['Monthly Return']
            df.to_excel(writer, sheet_name=portfolio_name)


def compute_performance_stats(portfolio_returns):
    summary = {}
    for name, series in portfolio_returns.items():
        cumulative = (1 + series).cumprod()
        ann_return = (1 + series.mean()) ** 12 - 1
        ann_vol = series.std() * np.sqrt(12)
        sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan   # double check this
        drawdown = cumulative / cumulative.cummax() - 1  # double check this
        max_dd = drawdown.min()  # double check this

        summary[name] = {
            'Cumulative Return': cumulative.iloc[-1] - 1,
            'Annualized Return': ann_return,
            'Annualized Volatility': ann_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd
        }
    
    df = pd.DataFrame(summary).T
    df.to_csv("portfolios/performance_stats.csv", index=True)
    return df


def plot_portfolio_performance(portfolio_returns):
    cumulative = (1 + portfolio_returns).cumprod()
    drawdowns = cumulative.div(cumulative.cummax()) - 1

    plt.figure(figsize=(14, 6))
    sns.lineplot(data=cumulative)
    plt.title("Cumulative Portfolio Returns", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("portfolio_performance_plots/cumulative_returns.png")
    plt.close()

    for portfolio in portfolio_returns.columns:
        plt.figure(figsize=(14, 6))
        sns.barplot(x=portfolio_returns.index, y=portfolio_returns[portfolio])
        plt.xticks(rotation=45)
        plt.title(f"Monthly Returns: {portfolio}", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("Monthly Return")
        plt.tight_layout()
        plt.savefig(f"portfolio_performance_plots/{portfolio}_monthly_returns.png")
        plt.close()

    plt.figure(figsize=(14, 6))
    sns.lineplot(data=drawdowns)
    plt.title("Portfolio Drawdowns", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("portfolio_performance_plots/drawdown_chart.png")
    plt.close()

    plt.figure(figsize=(10, 7))
    ann_return = (1 + portfolio_returns.mean()) ** 12 - 1
    ann_vol = portfolio_returns.std() * np.sqrt(12)
    sns.scatterplot(x=ann_vol, y=ann_return, s=100)
    for name in portfolio_returns.columns:
        plt.text(ann_vol[name], ann_return[name], name, fontsize=12, weight='bold')
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.title("Risk-Return Profile", fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("portfolio_performance_plots/risk_return_scatter.png")
    plt.close()


def summarize_portfolios(portfolios, rolling_betas):
    for name, weights in portfolios.items():
        weights = weights.fillna(0).astype(float)  
        weights.to_csv(f"portfolio_summaries/{name}_weights.csv")

        included = weights[weights > 0].count()
        included = included.sort_values(ascending=False).rename("InclusionCount")
        avg_weights = weights.mean().rename("AverageWeight")
        summary = pd.concat([included, avg_weights], axis=1)
        summary.to_csv(f"portfolio_summaries/{name}_summary.csv")

        avg_betas_list = []
        for date in weights.index:
            current_weights = weights.loc[date]
            nonzero = current_weights[current_weights > 0]
            if nonzero.empty:
                continue
            betas = pd.DataFrame({f: rolling_betas[f].loc[date] for f in nonzero.index if date in rolling_betas[f].index}).T
            avg_betas_list.append(betas.mean())

        if avg_betas_list:
            avg_macro = pd.concat(avg_betas_list, axis=1).mean(axis=1).to_frame(name="AverageMacroBeta")
            avg_macro.to_csv(f"portfolio_summaries/{name}_macro_summary.csv")

        turnover = weights.diff().abs().sum(axis=1)
        turnover_df = pd.DataFrame({"Turnover": turnover})
        turnover_df.to_csv(f"portfolio_summaries/{name}_turnover.csv")
        
        heatmap_data = weights.copy()
        heatmap_data['Month'] = heatmap_data.index.strftime('%b')
        heatmap_data['Year'] = heatmap_data.index.year
        heatmap_data = heatmap_data.set_index(['Year', 'Month'])
        heatmap_pivot = heatmap_data.groupby(['Year', 'Month']).mean().T

        plt.figure(figsize=(14, 6))
        sns.heatmap(heatmap_pivot, cmap="viridis", cbar=True, linewidths=0.5)
        plt.title(f"{name} Portfolio Weight Heatmap", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("Funds")
        plt.tight_layout()
        plt.savefig(f"portfolio_summaries/{name}_weight_heatmap.png")
        plt.close()




if __name__ == '__main__':
    file_path = 'fund_list.csv' 
    start_date = '2015-01-01'
    end_date = '2024-12-31'

    monthly_returns = monthly_returns(file_path,start_date,end_date)
    # monthly_returns.to_csv("monthly_returns.csv", index=False)
    tickers = monthly_returns.columns
    
    macro_var_std = macro_variables(start_date,end_date)
    # macro_var_std.to_csv("macro.csv", index=False)
    
    ols_results = ols(monthly_returns,macro_var_std)
    
    rolling_results = rolling_reg(monthly_returns,macro_var_std)
    save_rolling_results_to_excel(rolling_results)
    
    plot_rolling_betas(rolling_results)

    # portfolios
    portfolios = construct_portfolios(rolling_results,tickers)
    portfolio_returns = simulate_portfolio_returns(portfolios,monthly_returns)
    save_portfolio_returns_to_excel(portfolio_returns)
    performance_stats = compute_performance_stats(portfolio_returns)

    plot_portfolio_performance(portfolio_returns)
    summarize_portfolios(portfolios,rolling_results)






