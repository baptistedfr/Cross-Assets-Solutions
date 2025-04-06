import numpy as np
import pandas as pd
from backtester272.Strategy import Strategy, OptimizationStrategy, filter_with_signals
from tqdm import tqdm


    
class MaxSharpeStrategy(OptimizationStrategy):
    """
    Stratégie d'optimisation maximisant le ratio de Sharpe.
    """
    def objective_function(self, weights: np.ndarray, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> float:
        """
        Fonction objectif pour maximiser le ratio de Sharpe du portefeuille.

        Args:
            weights (np.ndarray): Poids du portefeuille.
            expected_returns (pd.Series): Rendements attendus des actifs.
            cov_matrix (pd.DataFrame): Matrice de covariance.

        Returns:
            float: Négatif du ratio de Sharpe (pour minimisation).
        """
        portfolio_return = np.dot(weights, expected_returns) * 252  # Rendement annualisé
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)) * 252)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility

        # Terme de régularisation Ridge (si max_turnover est défini)
        if self.max_turnover is not None and hasattr(self, "current_position") and np.sum(np.abs(self.current_position)) > 0:
            ridge_penalty = self.lmd_ridge * np.sum((weights - self.current_position) ** 2)  # L2 penalty
        else:
            ridge_penalty = 0

        # Maximiser Sharpe => Minimiser son opposé + pénalité
        return -sharpe_ratio + ridge_penalty

