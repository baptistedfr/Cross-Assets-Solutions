import pandas as pd
import numpy as np
from datetime import datetime
from .DataMacro import DataMacro
from typing import Tuple


class MacroTactical():
    """
    Tilt des poids macro.
    """
    def __init__(self, alpha: float, window_size: float, threshold: float) -> None:
        """
        Initialise la stratégie avec le nom de la classe fille.
        """
        self.name: str = self.__class__.__name__
        self.alpha = alpha
        self.window_size = window_size
        self.threshold = threshold
        self.data_macro = DataMacro()

    def generate_window_data(self, current_date: datetime) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Retourne les données macro sur la bonne fenêtre.

        Args:
            current_date (datetime): date de rebalancement

        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: données macro sur la bonne fenêtre
        """
        start_date = current_date - pd.Timedelta(days=self.window_size)

        filtered_spread_2Y10Y = self.data_macro.spread_2Y10Y.loc[start_date:current_date].squeeze()
        filtered_yield_10Y = self.data_macro.yield_10Y.loc[start_date:current_date].squeeze()
        filtered_cpi = self.data_macro.cpi.loc[start_date:current_date].squeeze()
        filtered_ZEW = self.data_macro.ZEW.loc[start_date:current_date].squeeze()
        filtered_unemployment = self.data_macro.unemployment.loc[start_date:current_date].squeeze()
        filtered_savings = self.data_macro.savings.loc[start_date:current_date].squeeze()
        filtered_GDP = self.data_macro.GDP.loc[start_date:current_date].squeeze()

        return filtered_spread_2Y10Y, filtered_yield_10Y, filtered_cpi, filtered_ZEW, filtered_unemployment, filtered_savings, filtered_GDP
    
    def calculate_zscore_signal(self, series: pd.Series) -> float:
        """
        Calcule le Z-score de la série et renvoie un signal basé sur l'écart significatif.

        Args:
            series (pd.Series): La série de données.
            threshold (float): Le seuil pour considérer un écart comme significatif (en écart-types).

        Returns:
            float: Le signal (écart par rapport à la moyenne ou 0 si pas significatif).
        """

        mean = series.mean()
        std_dev = series.std()

        if std_dev == 0:
            return 0.0

        zscore = (series.iloc[-1] - mean) / std_dev

        if abs(zscore) > self.threshold:
            return zscore
        else:
            return 0.0


    def get_views(self, current_date: datetime) -> pd.Series:
        """
        Méthode obligatoire pour déterminer la position actuelle.

        Args:
            current_date (datetime): date de rebalancement
            current_position (pd.Series): La position actuelle.

        Returns:
            pd.Series: La nouvelle position (poids) pour chaque actif.
        """
        # On récupère les données avant la date 'current_date' sur la période de la fenêtre
        spread_2Y10Y, yield_10Y, cpi, zew_signal, unemployment, savings, GDP = self.generate_window_data(current_date)

        # On calcule les signaux
        spread_2Y10Y_signal = self.calculate_zscore_signal(spread_2Y10Y)
        yield_10Y_signal = self.calculate_zscore_signal(yield_10Y)
        cpi_signal = self.calculate_zscore_signal(cpi)
        zew_signal = self.calculate_zscore_signal(zew_signal)
        unemployment_signal = self.calculate_zscore_signal(unemployment)
        savings_signal = self.calculate_zscore_signal(savings)
        GDP_signal = self.calculate_zscore_signal(GDP)

        # On crée un dictionnaire avec les signaux
        macro_signals = pd.Series({
            'Pente 10Y-2Y': spread_2Y10Y_signal,
            'Taux10Y': yield_10Y_signal,
            'CPI Rate': cpi_signal,
            'ZEW': zew_signal,
            'Chômage': unemployment_signal,
            'Taux d’épargne': savings_signal,
            'GDP and Forecast': GDP_signal       
        })

        # Convertir les singaux en tilts avec la matrice de sensibilité
        tilts = self.data_macro.compute_weighted_view(macro_view=macro_signals)

        return tilts
    
    def get_position(self, current_date: datetime, current_position: pd.Series) -> pd.Series:
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
        view_position = self.get_views(current_date)

        # Combinaison linéaire entre la position d'équilibre et la position basée sur les vues
        new_position = (1 - self.alpha) * current_position + self.alpha * view_position
        # Normalisation pour que la somme des poids soit égale à 1
        new_position = new_position / new_position.abs().sum()

        return new_position