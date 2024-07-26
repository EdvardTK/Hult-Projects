#%%
import numpy as np
import itertools
import bt
import pandas as pd
import yfinance as yf
from talib import MACD, BBANDS
import pandas_datareader.data as web

#Setting up training data date variables
start_date = '2020-03-01'


# Import data for Optimal Portfolio Strategy
tickers = ['META', 'PYPL', 'SPY']
data1 = bt.get(tickers, start=start_date)
data1.head()

# We will need the risk-free rate to get correct Sharpe Ratios 
riskfree =  bt.get('^IRX', start=start_date)
# Convert risk free from % to decimal
riskfree_rate = float(riskfree.mean()) / 100
# Print out the risk free rate to make sure it looks good
print(riskfree_rate)

optimal_portfolio = bt.Strategy('Buy', 
                       [bt.algos.RunEveryNPeriods(15, offset=10),
                       bt.algos.SelectAll(),
                       bt.algos.WeighMeanVar(lookback=pd.DateOffset(months=3),bounds=(0.01,0.4),
                                              rf=0.05),
                       bt.algos.Rebalance()])

Buy = bt.Backtest(optimal_portfolio, data1)
result = bt.run(Buy)
result.plot()
result.display()
result.plot_security_weights()

data2 = bt.get(tickers, start=start_date)

# define the length of the short and long averages
short = 52
long = 200

# Calculate moving average DataFrame using rolling.mean
sma_short = data2.rolling(short).mean()
sma_long = data2.rolling(long).mean()

target_weights = sma_long.copy()

# We want 1/N bet on each asset, and length of tickers is the number of assets N
magnitude = 1/len(tickers)
# Set appropriate target weights based on the position of the curves.
# Note that if sma_short crossed moving up, sma_short>sma_long is true
target_weights[sma_short > sma_long] =  magnitude
target_weights[sma_short <= sma_long] = -magnitude
EMA_cross = bt.Strategy('EMA_cross', [bt.algos.WeighTarget(target_weights),
                                    bt.algos.Rebalance()])

test_MA = bt.Backtest(EMA_cross, data2)
res_MA = bt.run(test_MA)
res_MA.plot()
res_MA.display()
res_MA.plot_security_weights()

# Download historical data
#data3 = {ticker: yf.download(ticker, start=start_date) for ticker in tickers}
data3 = web.get_data_yahoo(tickers, start=start_date)['Adj Close']

# Define parameter ranges for grid search
fast_periods = range(20, 50, 10)
slow_periods = range(60, 100, 30)
signal_periods = range(20, 30, 10)

# Initialize variables to store the best parameters and corresponding performance
best_sharpe_ratio = -np.inf
best_params = None
best_result = None

def calculate_sharpe_ratio(backtest_result):
    daily_returns = backtest_result.prices.pct_change().dropna()
    mean_return = daily_returns.mean()
    std_return = daily_returns.std()
    sharpe_ratio = mean_return / std_return * np.sqrt(252)
    return sharpe_ratio.mean()

# Perform grid search
for fast, slow, signal in itertools.product(fast_periods, slow_periods, signal_periods):
    if fast >= slow:
        continue  # Skip invalid parameter combinations

    macd_data = pd.DataFrame(index=data3.index)
    magnitude = 1 / len(tickers)
    target_weights2 = pd.DataFrame(index=macd_data.index, columns=macd_data.columns)

    for ticker in tickers:
        real = data3[ticker]
        macd_line, signal_line, hist_line = MACD(real, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        macd_data[ticker, 'macd_line'] = macd_line
        macd_data[ticker, 'signal_line'] = signal_line
        target_weights2[ticker] = (macd_line < signal_line) * 0
        target_weights2[ticker] = (macd_line > signal_line) * -magnitude

    macd_bbands_strategy = bt.Strategy('MACD_BBands', 
                                       [bt.algos.SelectAll(),
                                        bt.algos.SelectMomentum(0),
                                        bt.algos.WeighTarget(target_weights2),
                                        bt.algos.Rebalance()])
    backtests = bt.Backtest(macd_bbands_strategy, data3)
    res_MACD = bt.run(backtests)
    
    sharpe_ratio = calculate_sharpe_ratio(res_MACD)
    
    if sharpe_ratio > best_sharpe_ratio:
        best_sharpe_ratio = sharpe_ratio
        best_params = (fast, slow, signal)
        best_result = res_MACD

# Output the best parameters and plot the result
print(f"Best Parameters: Fastperiod={best_params[0]}, Slowperiod={best_params[1]}, Signalperiod={best_params[2]}")
print(f"Best Sharpe Ratio: {best_sharpe_ratio}")
best_result.plot()
best_result.display()
best_result.plot_security_weights()

# Print the results for each ticker
for ticker in tickers:
    print(f"Results for {data3}:")
    #backtests[data4].display()

# Master Strategy
master_strategy = bt.Strategy('master', [bt.algos.RunMonthly(),
                                bt.algos.SelectAll(),
                                bt.algos.WeighEqually(),
                                bt.algos.Rebalance()],
                    [optimal_portfolio, EMA_cross, macd_bbands_strategy])

data_combined = pd.concat([data1, data2, data3], axis=1)
data_combined = data_combined.loc[:, ~data_combined.columns.duplicated()]  # Remove duplicate columns

# create the backtest and run it
test = bt.Backtest(master_strategy, data_combined)
# create results so we can display and plot
results = bt.run(test)
results.plot()
results.display()
results.plot_security_weights()
# You can click on the DataFrame in Variable Explorer instead of looking on web
df_results_key = results.stats.assign()
# Use the stats and at attributes to get total return and save it
interesting_result_to_save = results.stats.at['total_return','master']
# Use the stats and at attributes to get daily sharpe and save it
another_interesting_result_to_save = results.stats.at['daily_sharpe','master']
# %%
