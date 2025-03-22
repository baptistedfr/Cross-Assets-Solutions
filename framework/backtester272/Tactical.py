from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import List, Dict
from backtester272.Strategy import filter_with_signals
from pypfopt import BlackLittermanModel, black_litterman, risk_models


class Tactical(ABC):
    """
    Classe abstraite pour définir une stratégie d'investissement.

    Chaque stratégie doit implémenter la méthode `get_position`, qui détermine
    les poids des actifs dans le portefeuille à partir des données historiques.
    """
    def __init__(self) -> None:
        """
        Initialise la stratégie avec le nom de la classe fille.
        """
        self.name: str = self.__class__.__name__

    @abstractmethod
    def get_position(self, historical_data: pd.DataFrame, current_position: pd.Series) -> pd.Series:
        """
        Méthode obligatoire pour déterminer la position actuelle.

        Args:
            historical_data (pd.DataFrame): Les données historiques.
            current_position (pd.Series): La position actuelle.

        Returns:
            pd.Series: La nouvelle position (poids) pour chaque actif.
        """
        pass

class BlackLittermanTactical(Tactical):
    """
    Classe qui implémente une approche Black-Litterman en se basant sur le vecteur
    de poids 'current_position' comme poids d'équilibre, et en y intégrant des vues.
    
    Le paramètre alpha définit l'intensité avec laquelle les vues influencent la position finale.
    """
    def __init__(self, alpha: float = 0.5) -> None:
        """
        Args:
            alpha (float): Poids relatif des vues dans l'ajustement de la position.
                           0 donne la position d'équilibre, 1 la position entièrement basée sur les vues.
        """
        super().__init__()
        self.alpha = alpha
        self.cov_matrix = None

    def black_litterman_tilt(self, historical_data: pd.DataFrame, prior_weights: pd.Series, tactical_view: pd.Series):
        """
        Calcule le tilt des poids entre le prior (pondération stratégique) & les vues issues de la vue tactique (alphas détectés)
        à l'aide du modèle Black Litterman.

        Args:
            historical_data (pd.DataFrame): Les données historiques.
            prior_weights (pd.Series): Le vecteur de poids actuel
            tactical_view (pd.Series): Vues de l'allocation tactique
            
        Returns:
            pd.Series: La nouvelle position ajustée via Black Litterman
        """
        # Calcule la matrice de variance-covariance des rendements si elle n'a pas encore été définie
        if self.cov_matrix is None:
            self.cov_matrix = risk_models.sample_cov(historical_data, frequency=252)

        # Instancie un modèle Black-Litterman et calcule les nouveaux poids
        bl = BlackLittermanModel(cov_matrix=self.cov_matrix, pi=prior_weights, absolute_views=tactical_view)
        bl_weights = bl.bl_returns()
        bl_weights = bl_weights / bl_weights.abs().sum()

        # On met en forme les infos du prior, posterior et les vues pour voir si BL fonctionne comme on le souhaite
        rets_df = pd.DataFrame([prior_weights, bl_weights, tactical_view], index=["Prior", "Posterior", "Views"]).T
        rets_df.plot.bar(figsize=(12,8))

        return bl_weights
    
    def get_position(self, historical_data: pd.DataFrame, current_position: pd.Series) -> pd.Series:
        """
        Calcule la nouvelle position en combinant la position actuelle et les vues issues de get_views.
        
        Args:
            historical_data (pd.DataFrame): Les données historiques.
            current_position (pd.Series): Le vecteur de poids actuel (position d'équilibre).
            
        Returns:
            pd.Series: La nouvelle position ajustée.
        """

        if current_position.abs().sum() == 0:
            return current_position
        
        # Récupération des vues spécifiques à la stratégie
        view_position = self.get_views(historical_data, current_position)

        # # Combinaison linéaire entre la position d'équilibre et la position basée sur les vues
        # new_position = (1 - self.alpha) * current_position + self.alpha * view_position
        # # Normalisation pour que la somme des poids soit égale à 1
        # new_position = new_position / new_position.abs().sum()

        # Calcul des poids tiltés avec Black-Litterman
        new_position = self.black_litterman_tilt(historical_data=historical_data, prior_weights=current_position, tactical_view=view_position)

        return new_position

    @abstractmethod
    def get_views(self, historical_data: pd.DataFrame, current_position: pd.Series) -> pd.Series:
        """
        Méthode abstraite pour obtenir les vues de la stratégie.
        
        Ces vues doivent être retournées sous forme d'un vecteur de poids normalisé.
        
        Args:
            historical_data (pd.DataFrame): Les données historiques.
            current_position (pd.Series): Le vecteur de poids actuel.
            
        Returns:
            pd.Series: Les vues sous forme de vecteur de poids.
        """
        pass
