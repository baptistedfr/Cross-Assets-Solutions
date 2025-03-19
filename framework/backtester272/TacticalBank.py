import numpy as np
import pandas as pd
from backtester272.Tactical import BlackLittermanTactical, filter_with_signals

class MomentumTactical(BlackLittermanTactical):
    """
    Classe qui implémente la partie tactique Black-Litterman basée sur des signaux momentum.
    
    Les vues sont calculées à partir d'un signal momentum dérivé de l'historique des prix.
    """
    def __init__(self, delta: int = 0, **kwargs) -> None:
        """
        Initialise la tactique Momentum avec un paramètre pour définir le delta.

        Args:
            delta (int): Nombre de jours soustrait du jour T. On calcule le rendement entre T - (len(historical_data) - 1) et T - delta.
        """
        super().__init__(**kwargs)
        self.delta = delta
        
    @filter_with_signals
    def get_views(self, historical_data: pd.DataFrame, current_position: pd.Series) -> pd.Series:
        """
        Calcule les vues en se basant sur le momentum des actifs.
        
        Pour chaque actif, le rendement est calculé entre le premier prix de la série
        et le prix à T - delta (c'est-à-dire la (delta+1)-ème dernière observation). 
        Ce rendement est ensuite normalisé pour obtenir un vecteur de poids.
        
        Args:
            historical_data (pd.DataFrame): DataFrame contenant les prix historiques.
                                             Les colonnes représentent les actifs.
            current_position (pd.Series): Le vecteur de poids actuel (non utilisé ici, mais disponible si besoin).
            
        Returns:
            pd.Series: Vecteur de poids basé sur le signal momentum.
        """
        # Prix de départ : première observation de la série
        start_prices = historical_data.iloc[0]
        # Prix à T - delta : la (delta+1)-ème dernière observation
        end_prices = historical_data.iloc[-(self.delta + 1)]
        # Calcul du rendement momentum
        momentum = (end_prices / start_prices) - 1
        # Normalisation pour obtenir un vecteur de poids
        normalized_views = momentum / momentum.abs().sum()
        return normalized_views
    
class ValueTactical(BlackLittermanTactical):
    """
    Classe qui implémente la partie tactique Black-Litterman basée sur un signal de value.
    
    Le signal est calculé comme le rapport de l'ancien prix sur le nouveau prix.
    """

    @filter_with_signals
    def get_views(self, historical_data: pd.DataFrame, current_position: pd.Series) -> pd.Series:
        """
        Calcule les vues en se basant sur un signal value.
        
        Pour chaque actif, le signal est calculé comme le rapport de l'ancien prix (première observation)
        sur le nouveau prix (la (delta+1)-ème dernière observation). 
        Ce signal est ensuite normalisé pour obtenir un vecteur de poids.
        
        Args:
            historical_data (pd.DataFrame): DataFrame contenant les prix historiques.
                                             Les colonnes représentent les actifs.
            current_position (pd.Series): Le vecteur de poids actuel (non utilisé ici, mais disponible si besoin).
            
        Returns:
            pd.Series: Vecteur de poids basé sur le signal value.
        """
        # Prix de départ : première observation de la série
        start_prices = historical_data.iloc[0]
        # Prix à T - delta : la (delta+1)-ème dernière observation
        end_prices = historical_data.iloc[-1]
        # Calcul du signal value : ancien prix / nouveau prix
        value_signal = start_prices / end_prices
        # Normalisation pour obtenir un vecteur de poids
        normalized_views = value_signal / value_signal.abs().sum()
        return normalized_views