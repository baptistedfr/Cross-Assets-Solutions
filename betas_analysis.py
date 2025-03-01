import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_excel("datas/Stoxx600_sectors_prices_cleen.xlsx", index_col=0)
market = pd.read_excel("datas/stoxx600.xlsx", index_col=0)

returns_df = data.pct_change().dropna()
market_return = market.pct_change().dropna()
returns_df, market_return = returns_df.align(market_return, join="inner", axis=0)

var_rm = market_return.rolling(window=126).var()
beta_df = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
for ticker in returns_df.columns:
    # Covariance entre l'actif et le marché sur la fenêtre glissante
    cov_series = returns_df[ticker].rolling(window=126).cov(market_return)
    beta_df[ticker] = cov_series / var_rm
beta_df = beta_df.dropna()


fig = go.Figure()

for column in beta_df.columns:
    fig.add_trace(go.Scatter(x=beta_df.index, y=beta_df[column], mode='lines', name=column))

fig.update_layout(title="Évolution des betas des secteurs dans le temps",
                  xaxis_title="Date",
                  yaxis_title="Beta",
                  legend_title="Secteurs",
                  hovermode="x")
fig.show()