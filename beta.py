import pandas as pd
import numpy as np
import plotly.express as px


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


p = pd.read_csv("stoxx_600_prix.csv", index_col=0, parse_dates=True)
p  = p.drop('VMUK LN Equity', axis=1) # Pas de valeur pour ce ticker
p = p.drop('LOOMIS SS Equity', axis=1) # Pas de valeur pour ce ticker

comp = pd.read_csv("stoxx_600_compo.csv", index_col=0, parse_dates=True)
sect = pd.read_csv("stoxx_600_secteurs.csv", index_col=0, parse_dates=True)
comp = comp.drop('VMUK LN Equity', axis=1, errors='ignore')
sect = sect.drop('VMUK LN Equity', axis=0, errors='ignore')
comp = comp.drop('LOOMIS SS Equity', axis=1, errors='ignore')
sect = sect.drop('LOOMIS SS Equity', axis=0, errors='ignore')
p_index = pd.read_excel("stoxx600.xlsx", index_col=0, parse_dates=True)
p_index = p_index.replace(',', '.', regex=True)
p_index = p_index.astype(float)

start = p.index[0]
end = p.index[-1]
rebalancing_dates = [d.date() for d in pd.date_range(start=start, end=end, freq='YE')]
all_betas = []

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
        print(f"Processing sector: {sector}")
        p_sector = p_bis[tickers]
        p_sector = p_sector.bfill()
        # tickers_with_na = p_sector.columns[p_sector.isna().sum() > 0].tolist()
        p_sector = p_sector.dropna()
        p_sector = p_sector.astype(float)
        p_sector_returns = p_sector.pct_change().dropna().mean(axis=1)
        try:
            sector_cumulative_returns[sector] = p_sector_returns.add(1).cumprod()
        except:
            print(f"Error for sector: {sector}")
            continue

    sector_betas = {'date': d}

    for sector, cumulative_return in sector_cumulative_returns.items():
        aligned_returns = pd.concat([cumulative_return, market_cumulative_return], axis=1).dropna()

        if aligned_returns.empty:
            print(f"No overlapping data for sector: {sector}")
            continue

        sector_return = aligned_returns.iloc[:, 0]
        market_return = aligned_returns.iloc[:, 1]
        covariance = np.cov(market_return, sector_return)[0, 1]
        variance = np.var(sector_return)
        beta = covariance / variance
        sector_betas[sector] = beta

    all_betas.append(sector_betas)

# all_betas_df = pd.DataFrame(all_betas)
# all_betas_df.set_index('date', inplace=True)
# all_betas_df.to_csv("evolution_sector_betas_bis.csv")

# Assuming all_betas_df is already defined
all_betas_df = pd.read_csv("evolution_sector_betas_bis.csv", index_col='date', parse_dates=True)

# Convert the DataFrame to a long-form format
all_betas_long = all_betas_df.reset_index().melt(id_vars='date', var_name='Sector', value_name='Beta')

# Create the plot
fig = px.line(all_betas_long, x='date', y='Beta', color='Sector', title='Sector Betas Over Time')

# Show the plot
fig.show()

