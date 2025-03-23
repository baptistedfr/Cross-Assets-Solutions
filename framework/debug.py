import backtester272 as bt
import pandas as pd
import numpy as np
import os

# Print where this file is located
print(os.path.abspath(__file__))

# Set the directory to the data folder
os.chdir(os.path.dirname(__file__))

def load_data(name):
    data = pd.read_csv(f'data/{name}.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data.sort_index()
    return data


#benchmark = load_data('benchmark')
weights = load_data('weights')  
sectors = load_data('sectors')

wip = bt.Backtester(sectors, benchmark_weights=weights)

start_date = sectors.index[0].strftime('%Y-%m-%d')
end_date = sectors.index[-1].strftime('%Y-%m-%d')

global_params = {
    'start_date': start_date,
    'end_date': end_date,
    'freq': 30 * 6,
    'window': 30 * 6,
    'freq_tactical': 30,
    'window_tactical': 30 * 2,
    'aum': 100,
    'transaction_cost': 0
}

stategy_constraints = {
    'max_weight': 0.35,
    'min_weight': 0.025,
    'risk_free_rate': 0.02,
    'total_exposure': 1.0,
    'max_turnover': 1,
    'max_tracking_error': 0.05,
    'lmd_ridge': 0
}

tactical_constraints = {
    'alpha': 0.25,
    'delta': 30,
}

#1MSM = wip.run(**global_params, strategy=bt.MaxSharpeStrategy(**stategy_constraints), tactical=bt.MomentumTactical(**tactical_constraints))
MSRM = wip.run(**global_params, strategy=bt.MaxSharpeStrategy(**stategy_constraints), tactical=bt.RankMomentumTactical(**tactical_constraints))