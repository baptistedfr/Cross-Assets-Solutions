import numpy as np
import pandas as pd

class Backtest:
    def __init__(self, returns, market_returns, macro_data, num_portfolios=1000):
        """
        Initialize the Backtest class.

        :param returns: DataFrame of asset returns
        :param market_returns: Series of market returns (STOXX 600)
        :param macro_data: DataFrame of macroeconomic data
        :param num_portfolios: Number of random portfolios to generate
        """
        self.returns = returns
        self.market_returns = market_returns
        self.macro_data = macro_data
        self.num_portfolios = num_portfolios
        self.weights = None

    def mean_variance_allocation(self):
        """
        Perform mean-variance allocation by generating random portfolios and selecting the one with minimum variance.
        """
        num_assets = len(self.returns.columns)
        results = np.zeros((self.num_portfolios, num_assets + 2))

        for i in range(self.num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            portfolio_return = np.sum(weights * self.returns.mean()) * 252
            portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
            results[i, 0:num_assets] = weights
            results[i, num_assets] = portfolio_return
            results[i, num_assets + 1] = portfolio_stddev

        min_var_index = np.argmin(results[:, num_assets + 1])
        self.weights = results[min_var_index, 0:num_assets]
        return self.weights

    def adjust_allocation(self, stress_levels):
        """
        Adjust allocation based on macroeconomic stress levels.

        :param stress_levels: Series of stress levels
        """
        # Example adjustment: reduce allocation proportionally to stress level
        adjustment_factor = 1 - (stress_levels.iloc[-1] / 4)  # Use the latest stress level
        adjusted_weights = self.weights * adjustment_factor
        self.weights = adjusted_weights / adjusted_weights.sum()  # Normalize weights
        return self.weights

    def run_backtest(self):
        """
        Run the backtest with mean-variance allocation and macro adjustment.
        """
        self.mean_variance_allocation()
        stress_levels = self.macro_data['stress_level']
        self.adjust_allocation(stress_levels)
        return self.weights