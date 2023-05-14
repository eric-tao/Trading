import numpy as np
from statistics import median

win_threshold = 0.532

risk_tol = 0.03
buy_lim = 250

start_cash = 100000
iterations = 10000
rounds = 250
outcomes = []

for i in range(0,iterations):
    cash = start_cash
    generator = np.random.default_rng()
    for j in range(0,rounds):
        contracts = round(cash*risk_tol/buy_lim)
        if contracts < 1:
            break
        outcome = generator.random()
        if outcome < win_threshold:
            cash += contracts * (500 - buy_lim)
        else:
            cash -= contracts * buy_lim
    outcomes.append(cash)

winners = list(filter(lambda x: x > start_cash, outcomes))
loser   = list(filter(lambda x: x < start_cash, outcomes))

print(f"Median: {median(outcomes)}")
print(f"Max: {max(outcomes)}")
print(f"Min: {min(outcomes)}")
print(f"Losing in {len(loser)} outcomes")
print(f"Winning in {len(winners)} outcomes")
