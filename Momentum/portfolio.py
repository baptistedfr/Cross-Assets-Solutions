from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
import numpy as np
from scipy.stats import zscore
from scipy.optimize import linprog
from sklearn.covariance import LedoitWolf
from enum import Enum
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns

class Weighting_Type(Enum):
    EQUAL_WEIGHT = "EW"

class BasePortfolio(ABC):
    """Classe de base pour les portefeuilles factoriels"""

    def __init__(self, df_factor: pd.DataFrame, 
                 target_factor: str):
        """
        Args:
            df_factor (pd.DataFrame): Données factorielles avec colonnes ['Date', 'Ticker', facteurs...]
            target_factor (str): Facteur à maximiser
            sensi_factors (list[str]): Liste des facteurs pour calculer les sensibilités
            weighting_type (WeightingType): Type de pondération
        """
        self.df_factor = df_factor
        self.target_factor = target_factor

        # Vérifications
        required_cols = {'Date', 'Ticker'} | {target_factor}
        if not required_cols.issubset(df_factor.columns):
            raise ValueError(f"Le DataFrame doit contenir les colonnes suivantes : {required_cols}")


    @abstractmethod
    def apply_weights(self, df_ptf: pd.DataFrame) -> pd.DataFrame:
        """Calcule le poids des actifs selon le portfolio type"""
        pass

    def construct_portfolio(self, df_factor: pd.DataFrame, rebalance_weight: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """Construit un portefeuille fractile"""
        df_ptf = self.compute_zscore(df_factor)
        if rebalance_weight:
            df_ptf = self.apply_weights(df_ptf)
        else:
            if 'Weight' not in df_ptf.columns:
                raise ValueError("Please Provide a Weight column")

        return df_ptf

    def compute_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les z-scores"""
        df[f'Zscore_{self.target_factor}'] = zscore(df[self.target_factor], nan_policy='omit').fillna(0)
        return df



class FractilePortfolio(BasePortfolio):
    """Portefeuille basé sur une approche de classement en fractiles"""

    def __init__(self, df_factor: pd.DataFrame, 
                 target_factor: str, 
                 nb_fractile: int = 4, 
                 weighting_type: Weighting_Type = Weighting_Type.EQUAL_WEIGHT):
        
        super().__init__(df_factor, target_factor)
        self.nb_fractile = nb_fractile
        self.weighting_type = weighting_type
        if nb_fractile < 2:
            raise ValueError("Le nombre de fractiles doit être au moins 2.")


    def apply_weights(self, df_ptf: pd.DataFrame) -> pd.DataFrame:
        """Applique la pondération long-short sur le portefeuille ainsi que le calcul des fractiles"""
        
        df_ptf[f'Fractile_{self.target_factor}'] = pd.qcut(df_ptf[f'Zscore_{self.target_factor}'], q=self.nb_fractile, labels=range(1, self.nb_fractile + 1))
    
        num_long = len(df_ptf[df_ptf[f"Fractile_{self.target_factor}"] == self.nb_fractile])
        num_short = len(df_ptf[df_ptf[f"Fractile_{self.target_factor}"] == 1])

        # if num_long == 0 or num_short == 0:
        #     raise ValueError("Un des fractiles est vide, impossible d'appliquer les poids.")
        df_ptf['Weight'] = df_ptf[f"Fractile_{self.target_factor}"].apply(
            lambda x: 1 / num_long if x == self.nb_fractile else 0
        )
        # df_ptf['Weight'] = df_ptf[f"Fractile_{self.target_factor}"].apply(
        #     lambda x: 1 / num_long if x == self.nb_fractile else -1 / num_short if x == 1 else 0
        # )
        return df_ptf

