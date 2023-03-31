import numpy as np
import pandas as pd
from numba import njit
import pybroker
from pybroker import Strategy, YFinance, IndicatorSet
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import r2_score

pybroker.enable_data_source_cache('yfinance')

yfinance = YFinance()

def sma(bar_data, lookback):

    @njit  # Enable Numba JIT.
    def vec_sma(values):
        # Initialize the result array.
        n = len(values)
        out = np.array([np.nan for _ in range(n)])

        # For all bars starting at lookback:
        for i in range(lookback, n):
            # Calculate the moving average for the lookback.
            ma = 0
            for j in range(i - lookback, i):
                ma += values[j]
            ma /= lookback
            # Subtract the moving average from value.
            out[i] = ma
        return out

    # Calculate with close prices.
    return vec_sma(bar_data.close)

sma_8 = pybroker.indicator('sma_8', sma, lookback=8)
sma_13 = pybroker.indicator('sma_13', sma, lookback=13)
sma_21 = pybroker.indicator('sma_21', sma, lookback=21)

def diff_sma(bar_data, first,second):

    @njit  # Enable Numba JIT.
    def vec_sma(values,lookback):
        # Initialize the result array.
        n = len(values)
        out = np.array([np.nan for _ in range(n)])

        # For all bars starting at lookback:
        for i in range(lookback, n):
            # Calculate the moving average for the lookback.
            ma = 0
            for j in range(i - lookback, i):
                ma += values[j]
            ma /= lookback
            # Subtract the moving average from value.
            out[i] = ma
        return out

    sma_8  = vec_sma(bar_data.close,8)
    sma_21 = vec_sma(bar_data.close,21)
    # Calculate with close prices.
    return sma_8 - sma_21

sma_diff_8_21 = pybroker.indicator('sma_diff_8_21', diff_sma, first=8, second = 21)

def buy_sma_cross(ctx):
    if ctx.long_pos():
        return
    if ctx.indicator('sma_diff_8_21')[-1] > 0:
        ctx.buy_shares = ctx.calc_target_shares(0.5)
        ctx.hold_bars = 5
    #if ctx.indicator('sma_8')[-1] > ctx.indicator('sma_21')[-1]:
    #    ctx.buy_shares = ctx.calc_target_shares(0.5)
    #    ctx.hold_bars = 5

#strategy = Strategy(yfinance, '3/1/2021', '3/1/2023')
#strategy.add_execution(buy_sma_cross, 'AAPL', indicators=[sma_diff_8_21])
#pybroker.enable_indicator_cache('my_indicators')
#result = strategy.backtest(calc_bootstrap=False)
#result.metrics_df
#print(result.metrics_df)
pybroker.enable_caches('walkforward_strategy')

def train_slr(symbol, train_data, test_data):
    # Train
    # Previous day close prices.
    train_prev_close = train_data['close'].shift(1)
    # Calculate daily returns.
    train_daily_returns = (train_data['close'] - train_prev_close) / train_prev_close
    # Predict next day's return.
    train_data['pred'] = train_daily_returns.shift(-1)
    train_data = train_data.dropna()
    # Train the LinearRegession model to predict the next day's return
    # given the 8 against 21-day SMA.
    #train_diff = train_data[['sma_8']] - train_data[['sma_21']].values
    #x_train = train_data[['sma_8','sma_21']]
    #x_train = train_data[['sma_21']]
    x_train = train_data[['sma_diff_8_21']]
    y_train = train_data[['pred']]
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Test
    test_prev_close = test_data['close'].shift(1)
    test_daily_returns = (test_data['close'] - test_prev_close) / test_prev_close
    test_data['pred'] = test_daily_returns.shift(-1)
    #test_diff = test_data[['sma_8']] - test_data[['sma_21']].values
    #x_test = test_data[['sma_8','sma_21']]
    #x_test = test_data[['sma_21']]
    x_test = test_data[['sma_diff_8_21']]
    y_test = test_data[['pred']]
    # Make predictions from test data.
    y_pred = model.predict(x_test)
    # Print goodness of fit.
    #r2 = r2_score(y_test, np.squeeze(y_pred))
    #print(symbol, f'R^2={r2}')

    # Return the trained model.
    return model

sma_indicators = IndicatorSet()
sma_indicators.add(sma_8, sma_21)

model_slr = pybroker.model('slr', train_slr, indicators=[sma_diff_8_21])
training_strategy = Strategy(YFinance(), '3/1/2015', '3/1/2020')
training_strategy.add_execution(None, ['SPY'], models=model_slr)
result2 = training_strategy.backtest(train_size=0.25)

def hold_long(ctx):
    if not ctx.long_pos():
        # Buy if the next bar is predicted to have a positive return:
        if ctx.preds('slr')[-1] > 0:
            ctx.buy_shares = ctx.calc_target_shares(0.25)
    else:
        # Sell if the next bar is predicted to have a negative return:
        if ctx.preds('slr')[-1] < 0:
            ctx.sell_all_shares()


future_strategy = Strategy(YFinance(),'3/1/2020', '3/1/2023')
future_strategy.add_execution(hold_long, ['SPY'], models=model_slr)
future_results = future_strategy.walkforward(windows=3, train_size=0.5, lookahead=1)
print(future_results.metrics_df)
