import pybroker
import numpy  as np
import pandas as pd
import random
from sklearn  import svm
from pybroker import YFinance
from statistics import median

def vec_sma(values, lookback):
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

def rsi(df, periods = 14, ema = True):
    close_delta = df['close'].diff()
    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    if ema == True:
	    # Use exponential moving average
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window = periods, adjust=False).mean()
        ma_down = down.rolling(window = periods, adjust=False).mean()
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi

pybroker.enable_data_source_cache('yfinance')
yfinance = YFinance()

start_date = "1/1/2009"
end_date = "1/1/2023"

vix_df       = yfinance.query('^VIX', start_date=start_date, end_date=end_date)
gspc_df      = yfinance.query('^GSPC', start_date=start_date, end_date=end_date)
gspc_5_sma_list   = pd.DataFrame(data=vec_sma(gspc_df.close, 5)).fillna(0)[0].tolist()
gspc_8_sma_list   = pd.DataFrame(data=vec_sma(gspc_df.close, 8)).fillna(0)[0].tolist()
gspc_13_sma_list  = pd.DataFrame(data=vec_sma(gspc_df.close, 13)).fillna(0)[0].tolist()
gspc_200_sma_list = pd.DataFrame(data=vec_sma(gspc_df.close, 200)).fillna(0)[0].tolist()
rsi_list          = rsi(gspc_df).tolist()
#gspc_200_sma = pd.DataFrame(gspc_df, 200)
gspc_labels  = np.array((gspc_df.close > gspc_df.open).to_list())

temp_indicators = np.array([gspc_5_sma_list, gspc_8_sma_list, gspc_13_sma_list, gspc_df.close.tolist(), gspc_df.high.tolist(), gspc_df.low.tolist(), vix_df.high.tolist(), vix_df.low.tolist(), vix_df.open.tolist(), vix_df.close.tolist(), rsi_list ])

columns = len(temp_indicators)
temp_indicators = temp_indicators.transpose()
rows = len(temp_indicators) - 1
#delete last row
temp_indicators = np.delete(temp_indicators,rows, axis=0)
#delete first row
opens = np.delete(gspc_df.open.tolist(),0)
indicators = np.empty((rows,columns+1))
for i in range(0,len(opens)):
    indicators[i] = np.append(temp_indicators[i],opens[i])

cutoff     = 14
indicators = indicators[cutoff:-1]
gspc_labels = gspc_labels[cutoff+1:-1]

model_results = {}
for train_proportion in np.linspace(0,0.7,25):
    if train_proportion == 0:
        continue
    sample_count = round(len(indicators)*train_proportion)
    #print(sample_count)
    acc_results = []

    for iterations in range(0,20):

        train_indices = []
        while len(train_indices) < sample_count:
            new_number = random.randrange(0,len(indicators))
            if new_number in train_indices:
                continue
            else:
                train_indices.append(new_number)

        train_indices.sort()
        train_rows = indicators[train_indices]
        train_labels = gspc_labels[train_indices]

        test_indices = list(set(range(0,len(indicators))) - set(train_indices))
        test_rows = indicators[test_indices]
        test_labels = gspc_labels[test_indices]

        model = svm.SVC(gamma='auto',C=0.5)
        model.fit(train_rows,train_labels)
        model.predict(test_rows)

        successes = np.where(model.predict(test_rows) == test_labels)[0]
        #print(f"This is iteration {iterations+1}")
        #print(len(successes))
        #print(len(test_rows))
        #print(len(successes)/len(test_rows))
        acc_results.append(len(successes)/len(test_rows))
    model_results[train_proportion] = { "median": median(acc_results), "max": max(acc_results), "min": min(acc_results)}

for key in model_results:
    print(f"For train proportion {key}:")
    print(f"Median: {model_results[key]['median']}")
    print(f"Max: {model_results[key]['max']}")
    print(f"Min: {model_results[key]['min']}")

