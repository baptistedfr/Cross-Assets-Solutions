import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from backtester272.Result import Result
from backtester272.Strategy import Strategy
 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Backtester:
    """
    Classe principale pour effectuer des backtests financiers.

    Cette classe permet d'exécuter des backtests sur des données de prix en utilisant différentes stratégies
    et configurations. Elle prend en charge les univers dynamiques (listes de tickers valides par date),
    la gestion des données manquantes, et le calcul de la performance du portefeuille.

    Attributes:
        data (pd.DataFrame): Données de prix pour les actifs.
        dates_universe (Dict[str, List[str]]): Univers des actifs par date, au format {date: [tickers]}.
        start_date (pd.Timestamp): Date de début du backtest.
        end_date (pd.Timestamp): Date de fin du backtest.
        freq (int): Fréquence de rééquilibrage en jours.
        window (int): Taille de la fenêtre de formation en jours.
        aum (float): Actifs sous gestion (AUM).
        transaction_cost (float): Coût de transaction en pourcentage.
        weights (pd.DataFrame): Poids des actifs calculés.
        performance (pd.Series): Performance du portefeuille sur la période du backtest.
        total_transaction_costs (float): Coût total des transactions pendant le backtest.
    """

    def __init__(self, data: pd.DataFrame, dates_universe: Dict[str, List[str]] = None, benchmark_weights: pd.DataFrame = None) -> None:
        """
        Initialise le backtester avec les données de prix et un univers d'actifs optionnel.

        Args:
            data (pd.DataFrame or pd.Series): Données de prix pour les actifs, indexées par date.
            dates_universe (Dict[str, List[str]], optional): Univers des actifs par date, au format
                {date: [tickers]}. Par défaut, aucun univers n'est défini.

        Raises:
            TypeError: Si les données ou `dates_universe` ne sont pas au bon format.
            ValueError: Si les données sont vides ou non quotidiennes.
        """
        # Gestion des données de prix
        if isinstance(data, pd.Series):
            self.data = pd.DataFrame(data)
            if self.data.columns[0] == 0:
                self.data.columns = ['Asset']
        else:
            self.data = data

        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Les données doivent être un DataFrame ou une Series.")

        if self.data.empty:
            raise ValueError("Les données ne peuvent pas être vides.")

        if (self.data.index.to_series().diff()[1:].dt.days < 1).all():
            raise ValueError("Les données doivent être quotidiennes.")
        
        if benchmark_weights is not None:
            # Vérifier si l'index est de type datetime
            if not isinstance(benchmark_weights.index, pd.DatetimeIndex):
                try:
                    benchmark_weights.index = pd.to_datetime(benchmark_weights.index)
                except ValueError:
                    raise ValueError("L'index des données doit être de type datetime.")

        self.benchmark_weights = benchmark_weights

        # Gestion de l'univers d'actifs par date
        self.dates_universe = {}
        if dates_universe is not None:
            if not isinstance(dates_universe, dict):
                raise TypeError("dates_universe doit être un dictionnaire.")

            for date_str, tickers in dates_universe.items():
                # Validation des clés et valeurs
                try:
                    pd.to_datetime(date_str)
                except ValueError:
                    raise ValueError(f"La clé {date_str} n'est pas une date valide au format YYYY-MM-DD.")
                
                if not isinstance(tickers, list) or not all(isinstance(t, str) for t in tickers):
                    raise ValueError(f"Les tickers pour la date {date_str} doivent être une liste de chaînes.")
                
                # Vérifie la présence des tickers dans les colonnes des données
                invalid_tickers = [t for t in tickers if t not in self.data.columns]
                if invalid_tickers:
                    raise ValueError(f"Tickers non trouvés dans les données: {invalid_tickers}.")
            
            self.dates_universe = dates_universe

    def run(self, 
            start_date: Optional[pd.Timestamp] = None, 
            end_date: Optional[pd.Timestamp] = None, 
            freq: int = 30, 
            window: int = 365, 
            aum: float = 100, 
            transaction_cost: float = 0.0, 
            strategy: Strategy = None) -> Result:
        """
        Exécute le backtest sur la période spécifiée avec les paramètres donnés.

        Args:
            start_date (pd.Timestamp, optional): Date de début du backtest. Par défaut, la première date des données.
            end_date (pd.Timestamp, optional): Date de fin du backtest. Par défaut, la dernière date des données.
            freq (int): Fréquence de rééquilibrage (en jours).
            window (int): Taille de la fenêtre de formation (en jours).
            aum (float): Actifs sous gestion (AUM).
            transaction_cost (float): Coût de transaction (en pourcentage).
            strategy (Strategy): Stratégie de trading à appliquer.

        Returns:
            Result: Objet contenant les résultats du backtest.

        Raises:
            ValueError: Si des paramètres obligatoires ou incohérents sont fournis.
        """
        if strategy is None:
            raise ValueError("Une stratégie doit être fournie pour exécuter le backtest.")

        if start_date is None:
            start_date = self.data.index[0]
        if end_date is None:
            end_date = self.data.index[-1]

        # Validation des paramètres
        if not isinstance(freq, int) or freq <= 0:
            raise ValueError("freq doit être un entier positif supérieur à 0.")
        if not isinstance(window, int) or window <= 0:
            raise ValueError("window doit être un entier positif supérieur à 0.")
        if not isinstance(aum, (int, float)) or aum <= 0:
            raise ValueError("aum doit être un float positif supérieur à 0.")
        if not isinstance(transaction_cost, (int, float)) or transaction_cost < 0:
            raise ValueError("transaction_cost doit être un float positif ou nul.")

        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.window = window
        self.aum = aum
        self.transaction_cost = transaction_cost

        # Gestion des données manquantes, calcul des poids et performance
        self.handle_missing_data()
        self.calculate_weights(strategy)
        self.performance = self.calculate_performance(self.weights)

        total_transaction_costs = self.total_transaction_costs

        if self.benchmark_weights is not None:
            self.benchmark = self.calculate_performance(self.benchmark_weights.loc[self.start_date:self.end_date].loc[self.weights.index[1]:])
        else:
            self.benchmark = None
        

        # Nommer la stratégie si elle n'est pas déjà nommée
        if not hasattr(strategy, 'name'):
            strategy.name = strategy.__class__.__name__

        return Result(self.performance, self.weights, total_transaction_costs, strategy.name, self.benchmark, self.benchmark_weights.loc[self.start_date:self.end_date].loc[self.weights.index[1]:])

    def handle_missing_data(self) -> None:
        """
        Gère les données manquantes en remplissant les valeurs dans les colonnes valides.
        """
        # Supprime les colonnes entièrement vides et conserve les données numériques
        self.data = self.data.dropna(axis=1, how='all').select_dtypes(include=[np.number])

        # Remplit les valeurs manquantes entre le premier et le dernier index valides
        for col in self.data.columns:
            self.data[col] = self.data[col].loc[
                self.data[col].first_valid_index():self.data[col].last_valid_index()
            ].ffill()

        if self.data.empty:
            raise ValueError("Aucune donnée disponible après le traitement des valeurs manquantes.")

    def calculate_weights(self, strategy: Strategy) -> None:
        """
        Calcule les poids optimaux pour chaque date de rééquilibrage en fonction de la stratégie.

        Args:
            strategy (Strategy): Stratégie de trading utilisée.
        """

        # Définir la fréquence de rééquilibrage et la fenêtre de formation en jours
        freq_dt = pd.DateOffset(days=self.freq)
        window_dt = pd.DateOffset(days=self.window)

        # Calculer la date de début en tenant compte de la fenêtre
        start_date_with_window = pd.to_datetime(self.start_date) - window_dt

        # Obtenir les données de prix sur la période pertinente
        prices = self.data[start_date_with_window:self.end_date]

        # Générer les dates de rééquilibrage en ordre décroissant
        rebalancing_dates = []
        current_date = prices.index[-1]
        while current_date >= prices.index[0] + window_dt:
            rebalancing_dates.append(current_date)
            current_date -= freq_dt

        # Inverser la liste pour avoir les dates en ordre croissant
        rebalancing_dates.reverse()

        # Initialiser les poids précédents à zéro pour tous les actifs
        last_weights = pd.Series(0.0, index=prices.columns)

        # Initialiser les listes pour collecter les poids et les dates
        weights_list = [last_weights]
        dates_list = [(current_date - pd.DateOffset(days=1))]

        # Calculer les poids pour chaque date de rééquilibrage
        for current_date in rebalancing_dates:
            # Définir la période de formation
            train_start = current_date - window_dt
            train_end = current_date - pd.DateOffset(days=1)

            # Obtenir les données de prix pour la période de formation
            price_window = prices[train_start:train_end]

            # Filtrer les données en fonction de l'univers défini
            if self.dates_universe:
                # Convertir les dates du dictionnaire en format datetime
                universe_dates = [pd.to_datetime(date) for date in self.dates_universe.keys()]

                # Trouver la date d'univers la plus récente avant la date courante
                available_dates = [date for date in universe_dates if date <= current_date]

                if available_dates:
                    reference_date = max(available_dates)
                    active_tickers = self.dates_universe[reference_date.strftime('%Y-%m-%d')]
                    price_window = price_window[active_tickers]
                else:
                    print(f"Pas d'univers défini avant {current_date}")
                    price_window = pd.DataFrame()  # Renvoie un DataFrame vide si aucune date valide

            # Supprimer les colonnes avec des valeurs manquantes
            price_window_filtered = price_window.dropna(axis=1)
            if price_window_filtered.empty:
                print(f"Aucune donnée disponible pour {current_date}. Passage...")
                continue

            # Calculer les nouveaux poids en fonction de la stratégie
            if self.benchmark_weights is not None:
                benchmark_weights = self.get_benchmark_weights(current_date)
                final_optimal_weights = strategy.get_position(price_window_filtered, last_weights, benchmark_weights)
            else:
                final_optimal_weights = strategy.get_position(price_window_filtered, last_weights)
            last_weights = final_optimal_weights

            # Enregistrer les poids et la date
            weights_list.append(final_optimal_weights)
            dates_list.append(current_date)

        # Créer un DataFrame à partir des poids collectés
        optimal_weights_df = pd.DataFrame(weights_list, index=dates_list)

        # Assigner les poids calculés à l'attribut de la classe
        self.weights = optimal_weights_df.fillna(0.0)

    def calculate_performance(self, weights: pd.Series) -> None:
        """
        Calcule la performance du portefeuille en utilisant les poids calculés.
        """
        
        # Initialiser le solde du portefeuille avec les actifs sous gestion (AUM)
        balance = self.aum

        # Obtenir la première date où des poids sont disponibles
        first_valid_date = weights.first_valid_index()

        # Filtrer les données de prix dans la plage de dates spécifiée
        df = self.data[self.start_date:self.end_date]

        # Calculer les rendements quotidiens
        returns = df.pct_change()[1:]

        # Initialiser les coûts totaux de transaction et les poids précédents
        self.total_transaction_costs = 0
        previous_weights = pd.Series(0.0, index=weights.columns)

        # Initialiser les listes pour stocker les valeurs du portefeuille et les dates
        portfolio_values = [self.aum]
        dates = [first_valid_date - pd.DateOffset(days=1)]

        # Obtenir la liste des dates à traiter
        date_range = returns.loc[first_valid_date:].index

        for date in date_range:
            # Mettre à jour les poids si de nouveaux poids sont disponibles
            if date in weights.index:
                current_weights = weights.loc[date]

                # Calculer les changements dans les positions
                changes = (current_weights - previous_weights) * balance

                # Calculer les coûts de transaction
                transaction_costs = changes.abs().sum() * (self.transaction_cost / 100)

                # Mettre à jour les coûts totaux de transaction et réduire le solde
                self.total_transaction_costs += transaction_costs
                balance -= transaction_costs

                # Mettre à jour les poids précédents
                previous_weights = current_weights.copy()
            else:
                current_weights = previous_weights.copy()

            # Calculer le rendement du portefeuille pour la journée
            portfolio_return = (current_weights * returns.loc[date]).sum()

            # Mettre à jour le solde
            balance *= (1 + portfolio_return)

            # Enregistrer la valeur du portefeuille et la date
            portfolio_values.append(balance)
            dates.append(date)

        # Créer une Series pour la performance du portefeuille
        return pd.Series(portfolio_values, index=dates)

    def get_benchmark_weights(self, date: pd.Timestamp) -> pd.Series:
        """
        Retourne les poids du benchmark pour une date donnée, en utilisant les poids les plus récents disponibles si la date exacte n'est pas présente.

        Args:
            date (pd.Timestamp): Date pour laquelle obtenir les poids.

        Returns:
            pd.Series: Poids du benchmark pour la date donnée.
        """
        if self.benchmark_weights is None:
            raise ValueError("Aucun poids de benchmark n'a été fourni.")

        if not isinstance(date, pd.Timestamp):
            date = pd.to_datetime(date)

        if date in self.benchmark_weights.index:
            return self.benchmark_weights.loc[date]
        else:
            valid_dates = self.benchmark_weights.index[self.benchmark_weights.index <= date]
            if valid_dates.empty:
                raise ValueError(f"Pas de poids de benchmark disponibles pour la date {date}.")
            closest_date = valid_dates.max()
            return self.benchmark_weights.loc[closest_date]
