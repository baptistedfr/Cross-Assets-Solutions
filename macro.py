import pandas as pd
import plotly.express as px
from utils.utils import MacroSensitivities
from backtest.backtest import Backtest

cpi = pd.read_csv("data_macro/cpi.csv", index_col=0, parse_dates=True)
gdp = pd.read_csv("data_macro/gdp_growth.csv", index_col=0, parse_dates=True)
unemployment = pd.read_csv("data_macro/unemployment_rate.csv", index_col=0, parse_dates=True)
ten_year_yield = pd.read_csv("data_macro/ten_year_bond_yield.csv", index_col=0, parse_dates=True)
ten_year_yield = ten_year_yield.iloc[6:]
# Assuming ten_year_yield is already defined
ten_year_yield.index = pd.to_datetime(ten_year_yield.index)
gdp = gdp.resample('M').last()
gdp = gdp.fillna(method='ffill')

# Merge all dataframes
macro_data = pd.concat([cpi, gdp, unemployment, ten_year_yield], axis=1)
macro_data.columns = ['CPI', 'GDP', 'Unemployment', '10Y Yield']

macro_data = macro_data.loc['2021-01-01':'2024-01-01']
macro_data = macro_data.resample('M').last()
macro_data = macro_data.fillna(method='ffill')
macro_data = macro_data.fillna(method='bfill')

seuil_taux = 2.0        # Taux supérieur à 2%
seuil_inflation = 3.0   # Inflation supérieure à 3%
seuil_pib_très_faible = 0.5  # Exemple : PIB inférieur à 1000 considéré comme très faible


# Appliquer la fonction sur chaque ligne de la DataFrame
macro_data['stress_level'] = macro_data.apply(MacroSensitivities.calc_stress_level, axis=1)

# Assuming macro_data is already defined
macro_data_long = macro_data.reset_index().melt(id_vars='index', var_name='Variable', value_name='Value')
macro_data_long.rename(columns={'index': 'Date'}, inplace=True)


# Create the plot
fig = px.line(macro_data_long, x='Date', y='Value', color='Variable', title='Macro Data Over Time')

# Show the plot
fig.show()

