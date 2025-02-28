import pandas as pd
import plotly.express as px
from utils.utils import MacroSensitivities
from backtest import Backtest

def closest(df, dt):
    """
    Computes the closest date from dt in a DataFrame index
    :param df: the DataFrame
    :param dt: the date from which we want the closest date
    :return: the closest date from dt in the DataFrame's index
    """
    try:
        dt = dt.date()
    except Exception:
        dt = dt
        pass

    try:
        return min(df.index, key=lambda x: abs(x - dt))
    except Exception:
        return min(df.index, key=lambda x: abs(x.date() - dt))

cpi = pd.read_csv("../data_macro/cpi.csv", index_col=0, parse_dates=True)
gdp = pd.read_csv("../data_macro/gdp_growth.csv", index_col=0, parse_dates=True)
unemployment = pd.read_csv("../data_macro/unemployment_rate.csv", index_col=0, parse_dates=True)
ten_year_yield = pd.read_csv("../data_macro/ten_year_bond_yield.csv", index_col=0, parse_dates=True)
ten_year_yield = ten_year_yield.iloc[6:]
ten_year_yield.index = pd.to_datetime(ten_year_yield.index)
gdp = gdp.resample('M').last()
gdp = gdp.fillna(method='ffill')

p = pd.read_csv("../datas/Tickers/stoxx_600_tickers_prix.csv", index_col=0, parse_dates=True)
p  = p.drop('VMUK LN Equity', axis=1) # Pas de valeur pour ce ticker
p = p.drop('LOOMIS SS Equity', axis=1) # Pas de valeur pour ce ticker

comp = pd.read_csv("../datas/Tickers/stoxx_600_compo.csv", index_col=0, parse_dates=True)
sect = pd.read_csv("../datas/Tickers/stoxx_600_secteurs.csv", index_col=0, parse_dates=True)
comp = comp.drop('VMUK LN Equity', axis=1, errors='ignore')
sect = sect.drop('VMUK LN Equity', axis=0, errors='ignore')
comp = comp.drop('LOOMIS SS Equity', axis=1, errors='ignore')
sect = sect.drop('LOOMIS SS Equity', axis=0, errors='ignore')
p_index = pd.read_excel("../datas/stoxx600.xlsx", index_col=0, parse_dates=True)
p_index = p_index.replace(',', '.', regex=True)
p_index = p_index.astype(float)

# Merge all dataframes
macro_data = pd.concat([cpi, gdp, unemployment, ten_year_yield], axis=1)
macro_data.columns = ['CPI', 'GDP', 'Unemployment', '10Y Yield']

macro_data = macro_data.loc['2021-01-01':'2024-01-01']
macro_data = macro_data.resample('M').last()
macro_data = macro_data.fillna(method='ffill')
macro_data = macro_data.fillna(method='bfill')

seuil_taux = 2.0
seuil_inflation = 3.0
seuil_pib_très_faible = 0.5

# Apply the function on each row of the DataFrame
macro_data['stress_level'] = macro_data.apply(MacroSensitivities.calc_stress_level, axis=1)

start = p.index[0]
end = p.index[-1]
rebalancing_dates = [d.date() for d in pd.date_range(start=start, end=end, freq='YE')]
all_betas = []
all_weights = []
aum = 1000000  # Initial AUM
performance = []
previous_weights = None

for date in rebalancing_dates[1:]:
    d = pd.to_datetime(date)
    p_date = p.loc[:closest(p, d)]
    comp_date = comp.loc[closest(p, d)].dropna()
    p_bis = p_date[comp_date.index]
    common_tickers = sect.index.intersection(p_bis.columns)
    sect_common = sect.loc[common_tickers]
    grouped_sect_common = sect_common.groupby('gics_sector_name').apply(lambda x: x.index.tolist()).to_dict()
    sector_cumulative_returns = {}
    market_cumulative_return = p_index.loc[start:d].mean(axis=1).pct_change().dropna().add(1).cumprod()

    for sector, tickers in grouped_sect_common.items():
        p_sector = p_bis[tickers]
        p_sector = p_sector.bfill()
        p_sector = p_sector.dropna()
        p_sector = p_sector.astype(float)
        p_sector_returns = p_sector.pct_change().dropna().mean(axis=1)
        try:
            sector_cumulative_returns[sector] = p_sector_returns.add(1).cumprod()
        except:
            continue

    # Perform backtest allocation
    returns = p_bis.pct_change().dropna()
    market_returns = p_index.loc[start:d].pct_change().dropna().mean(axis=1)
    macro_data_subset = macro_data.loc[:closest(macro_data, d)]

    backtest = Backtest(returns, market_returns, macro_data_subset)
    if previous_weights is not None:
        backtest.weights = previous_weights
    weights = backtest.run_backtest()

    portfolio_return = (weights * returns.iloc[-1]).sum()
    aum *= (1 + portfolio_return)
    performance.append({'date': d, 'aum': aum})
    previous_weights = weights  # Store current weights for next year's backtest


    all_weights.append({'date': d, 'weights': weights})


# Convert all_weights and performance to DataFrame for further analysis
weights_df = pd.DataFrame(all_weights)
weights_df.set_index('date', inplace=True)
weights_df.to_csv("../results/weights_backtest.csv")

performance_df = pd.DataFrame(performance)
performance_df.set_index('date', inplace=True)
performance_df.to_csv("../results/performance_backtest.csv")
print(performance_df)