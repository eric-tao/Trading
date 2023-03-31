import numpy as np
import pandas as pd
from numba import njit
import pybroker
from pybroker import Strategy, YFinance, StrategyConfig
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import r2_score



pybroker.enable_data_source_cache('yfinance')

def rsi_sma(bar_data):

    @njit
    def vec_sma(values,lookback):
        n = len(values)
        out = np.array([np.nan for _ in range(n)])
        for i in range(lookback, n):
            if np.isnan(values[i]):
                continue
            # Calculate the moving average for the lookback.
            ma = 0
            for j in range(i - lookback, i):
                ma += values[j]
            ma /= lookback
            out[i] = ma
        return out

    @njit  # Enable Numba JIT.
    def rsi_calc(open_data,close_data):
        # Initialize the result array.
        n = len(open_data)
        out = np.array([np.nan for _ in range(n)])

        # For all bars starting at lookback:
        for i in range(14, n):
            gains = 0
            gain_counter = 0
            losses = 0
            loss_counter = 0
            if i == 14:
                for j in range(0,14):
                    open = open_data[j]
                    close = close_data[j]
                    if close_data[j] >= open_data[j]:
                        gains += (close - open)/open
                        gain_counter += 1
                    else:
                        losses += (open - close)/open
                        loss_counter += 1

                average_gain = gains/gain_counter
                average_loss = losses/loss_counter
                out[i] = 100.0 - (100.0/(1.0 + average_gain/average_loss))
            else:
                for j in range(i-14,i):
                    open = open_data[j]
                    close = close_data[j]
                    if close_data[j] >= open_data[j]:
                        gains += (close - open)/open
                    else:
                        losses += (open - close)/open
                    
                current = (close_data[i] - open_data[i])/(open_data[i])
                if current > 0:
                    out[i] = 100.0 - (100.0/(1.0 + (gains + current)/losses))
                else:
                    out[i] = 100.0 - (100.0/(1.0 + gains/(losses - current)))
        ma = vec_sma(out,14)

        return ma

    return rsi_calc(bar_data.open,bar_data.close)

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

    # Calculate for close prices.
    return vec_sma(bar_data.close)

def train_model(symbol, train_data, test_data):
    # Train
    # Previous day close prices.
    train_prev_close = train_data['close'].shift(1)
    # Calculate daily returns.
    train_daily_returns = (train_data['close'] - train_prev_close) / train_prev_close
    # Predict next day's return.
    train_data['pred'] = train_daily_returns.shift(-1)
    train_data = train_data.dropna()
    # Train the LinearRegession model to predict the next day's return
    # given the 20-day CMMA.
    X_train = train_data[[ "rsi_ind", "sma_21", "sma_8"]]
    y_train = train_data[['pred']]
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Test
    test_prev_close = test_data['close'].shift(1)
    test_daily_returns = (test_data['close'] - test_prev_close) / test_prev_close
    test_data['pred'] = test_daily_returns.shift(-1)
    test_data = test_data.dropna()
    X_test = test_data[["rsi_ind","sma_21", "sma_8"]]
    y_test = test_data[['pred']]
    # Make predictions from test data.
    y_pred = model.predict(X_test)
    # Print goodness of fit.
    r2 = r2_score(y_test, np.squeeze(y_pred))
    print(symbol, f'R^2={r2}')

    # Return the trained model.
    return model

def hold_long(ctx):
    if not ctx.long_pos():
        # Buy if the next bar is predicted to have a positive return:
        if ctx.preds('slr')[-1] > 0:
            ctx.buy_shares = ctx.calc_target_shares(0.25)
    else:
        # Sell if the next bar is predicted to have a negative return:
        if ctx.preds('slr')[-1] < 0:
            ctx.sell_all_shares()

        


sma_8 = pybroker.indicator('sma_8', sma, lookback=8)
sma_21 = pybroker.indicator('sma_21', sma, lookback=21)
#rsi_ind = pybroker.indicator('rsi_ind', rsi_sma)

# define model and training
#model_slr = pybroker.model('slr', train_model, indicators=[rsi_ind,sma_21,sma_8])
#config = StrategyConfig(bootstrap_sample_size=100)

# define data set to use
#strategy = Strategy(YFinance(), '3/1/2014', '3/1/2019', config)
#strategy.add_execution(None, ['SPY'], models=model_slr)

# split into training and test sets, train
#strategy.backtest(train_size=0.5)

# test on test data
#strategy.clear_executions()
#strategy.add_execution(hold_long, ['SPY'], models=model_slr)
#result = strategy.walkforward(windows=3, train_size = 0.5, lookahead = 1)

#print(result.metrics_df.round(4))

def buy_8_21_cross(ctx):
    if ctx.long_pos():
        return
    if ctx.indicator('sma_8')[-1] > ctx.indicator('sma_21')[-1]:
        ctx.buy_shares = ctx.calc_target_shares(0.5)
        ctx.hold_bars = 2


dumb_strategy = Strategy(YFinance(), '3/1/2021', '3/1/2023')
dumb_strategy.add_execution(buy_8_21_cross, 'PLTR', indicators=[sma_21,sma_8])
result = dumb_strategy.backtest(calc_bootstrap=False)
print(result.metrics_df.round(4))