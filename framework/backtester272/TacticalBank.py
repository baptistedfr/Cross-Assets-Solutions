import numpy as np
import pandas as pd
from backtester272.Tactical import AlphaBasedTactical, filter_with_signals
from scipy.stats import zscore

class RankMomentumTactical(AlphaBasedTactical):
    """
    Classe qui implémente la partie tactique Black-Litterman basée sur des signaux momentum.
    
    Les vues sont calculées à partir d'un signal momentum dérivé de l'historique des prix.
    """
    def __init__(self, delta: int = 0, nb_fractile : int =4, **kwargs) -> None:
        """
        Initialise la tactique Momentum avec un paramètre pour définir le delta.

        Args:
            delta (int): Nombre de jours soustrait du jour T. On calcule le rendement entre T - (len(historical_data) - 1) et T - delta.
        """
        super().__init__(**kwargs)
        self.nb_fractile = nb_fractile
        self.delta = delta
        
    @filter_with_signals
    def get_views(self, historical_data: pd.DataFrame, current_position: pd.Series) -> pd.Series:
        """
        Calcule les vues en se basant sur le signal momentum et le fractionne en nb_fractile quantiles.
        On attribue un signal long (1) aux actifs du quantile supérieur et un signal short (-1)
        aux actifs du quantile inférieur.
        """
        # Prix de départ : première observation de la série
        start_prices = historical_data.iloc[0]
        # Prix à T - delta : la (delta+1)-ème dernière observation
        end_prices = historical_data.iloc[-(self.delta + 1)]
        # Calcul du rendement momentum
        momentum = (end_prices / start_prices) - 1

        # Fractionner le signal momentum en nb_fractile quantiles
        # Les labels vont de 0 (plus bas) à nb_fractile-1 (plus haut)
        quantiles = pd.qcut(momentum, q=self.nb_fractile, labels=False, duplicates='drop')

        # Initialiser un vecteur de vues à 0 pour tous les actifs
        views = pd.Series(0, index=momentum.index)
        # Attribuer 1 aux actifs du quantile supérieur (meilleur momentum)
        # views[quantiles == (self.nb_fractile - 1)] = 0
        # Attribuer -1 aux actifs du quantile inférieur (pire momentum)
        views[quantiles == 0] = -1

        
        normalized_views = momentum / momentum.abs().sum()
        #views = np.where(normalized_views > 0, 0, views)
        normalized_views = normalized_views * views

        return normalized_views