import pandas as pd
import plotly.express as px
from utils.utils import MacroSensitivities
from backtest.backtest import Backtest

cpi = pd.read_csv("data_macro/cpi.csv", index_col=0, parse_dates=True)
gdp = pd.read_csv("data_macro/gdp_growth.csv", index_col=0, parse_dates=True)
unemployment = pd.read_csv("data_macro/unemployment_rate.csv", index_col=0, parse_dates=True)
ten_year_yield = pd.read_csv("data_macro/ten_year_bond_yield.csv", index_col=0, parse_dates=True)
ten_year_yield = ten_year_yield.iloc[6:]
refi = pd.read_csv("data_macro/refinancing_rate.csv", index_col=0, parse_dates=True)
refi = refi.iloc[1:]
# Assuming ten_year_yield is already defined
ten_year_yield.index = pd.to_datetime(ten_year_yield.index)
gdp = gdp.resample('M').last()
gdp = gdp.fillna(method='ffill')

# Merge all dataframes
macro_data = pd.concat([cpi, gdp, unemployment, ten_year_yield, refi], axis=1)
macro_data.columns = ['CPI', 'GDP', 'Unemployment', '10Y Yield', 'Refi Rate']

macro_data = macro_data.loc['2021-01-01':'2024-01-01']
macro_data = macro_data.resample('M').last()
macro_data = macro_data.fillna(method='ffill')
macro_data = macro_data.fillna(method='bfill')


# Calcul de l'EMA pour chaque colonne (ici, sur 12 périodes par exemple)
ema_span = 12
for col in ['CPI', '10Y Yield', 'Refi Rate']:
    macro_data[f'{col}_ema'] = macro_data[col].ewm(span=ema_span, adjust=False).mean()

# On peut afficher un aperçu des données
print(macro_data.head())

# Appliquer la fonction sur chaque ligne de la DataFrame
macro_data['stress_level'] = macro_data.apply(MacroSensitivities.calc_stress_level, axis=1)
# macro_data.to_csv("results/macro_stress_data.csv")

# Assuming macro_data is already defined
macro_data_long = macro_data.reset_index().melt(id_vars='index', var_name='Variable', value_name='Value')
macro_data_long.rename(columns={'index': 'Date'}, inplace=True)


# Create the plot
fig = px.line(macro_data_long, x='Date', y='Value', color='Variable', title='Macro Data Over Time')

# Show the plot
fig.show()

