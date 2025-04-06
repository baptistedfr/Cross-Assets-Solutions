class RankedStrategy(Strategy):
    """
    Stratégie basée sur le classement des actifs en fonction de leurs caractéristiques
    ou performances. Les actifs avec les meilleurs rangs reçoivent les poids les plus élevés.
    """
    @filter_with_signals
    def get_position(self, historical_data: pd.DataFrame, current_position: pd.Series, benchmark_position: pd.Series = None) -> pd.Series:
        """
        Calcule la position (poids) des actifs en fonction de leur classement.

        Args:
            historical_data (pd.DataFrame): Les données historiques.
            current_position (pd.Series): La position actuelle.

        Returns:
            pd.Series: Poids normalisés basés sur le classement.
        """
        ranked_assets = self.rank_assets(historical_data)

        num_assets = ranked_assets.count()
        sum_of_ranks = ranked_assets.sum()
        average = sum_of_ranks / num_assets
        weights = (ranked_assets - average)

        total_abs_ranks = sum(abs(weights))
        normalized_weights = weights / total_abs_ranks

        return normalized_weights

    @abstractmethod
    def rank_assets(self, historical_data: pd.DataFrame) -> pd.Series:
        """
        Méthode abstraite pour classer les actifs.

        Args:
            historical_data (pd.DataFrame): Les données historiques.

        Returns:
            pd.Series: Classement des actifs (du meilleur au pire).
        """
        pass

class EqualWeightStrategy(Strategy):
    """
    Stratégie qui attribue un poids égal à chaque actif.
    """
    @filter_with_signals
    def get_position(self, historical_data: pd.DataFrame, current_position: pd.Series, benchmark_position: pd.Series = None) -> pd.Series:
        """
        Retourne une position avec des poids égaux pour chaque actif.

        Args:
            historical_data (pd.DataFrame): Les données historiques.
            current_position (pd.Series): La position actuelle.

        Returns:
            pd.Series: Nouvelle position avec des poids égaux.
        """
        num_assets = historical_data.shape[1]

        if num_assets == 0:
            return pd.Series()

        weights = pd.Series(1 / num_assets, index=historical_data.columns)
        return weights
    

class ShortEqualWeightStrategy(Strategy):
    """
    Stratégie qui attribue un poids négatif et égal à chaque actif.
    """
    @filter_with_signals
    def get_position(self, historical_data: pd.DataFrame, current_position: pd.Series, benchmark_position: pd.Series = None) -> pd.Series:
        """
        Retourne une position négative avec des poids égaux pour chaque actif.

        Args:
            historical_data (pd.DataFrame): Les données historiques.
            current_position (pd.Series): La position actuelle.

        Returns:
            pd.Series: Nouvelle position avec des poids égaux.
        """
        num_assets = historical_data.shape[1]

        if num_assets == 0:
            return pd.Series()

        weights = pd.Series(1 / num_assets, index=historical_data.columns)
        return - weights
    
class RandomStrategy(Strategy):
    """
    Stratégie qui attribue des poids aléatoires normalisés aux actifs.
    """
    @filter_with_signals
    def get_position(self, historical_data: pd.DataFrame, current_position: pd.Series, benchmark_position: pd.Series = None) -> pd.Series:
        """
        Retourne une position avec des poids aléatoires normalisés.

        Args:
            historical_data (pd.DataFrame): Les données historiques.
            current_position (pd.Series): La position actuelle.

        Returns:
            pd.Series: Nouvelle position avec des poids aléatoires.
        """
        weights = np.random.rand(len(historical_data.columns))
        weights /= weights.sum()  # Normaliser les poids pour qu'ils totalisent 1
        return pd.Series(weights, index=historical_data.columns)

class MinVarianceStrategy(OptimizationStrategy):
    """
    Stratégie d'optimisation minimisant la variance du portefeuille.
    """
    def objective_function(self, weights: np.ndarray, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> float:
        """
        Fonction objectif pour minimiser la variance du portefeuille.

        Args:
            weights (np.ndarray): Poids du portefeuille.
            expected_returns (pd.Series): Rendements attendus des actifs.
            cov_matrix (pd.DataFrame): Matrice de covariance.

        Returns:
            float: Variance du portefeuille.
        """
        #portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_variance = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)) * 252)

        # Terme de régularisation Ridge (si max_turnover est défini)
        if self.max_turnover is not None and hasattr(self, "current_position") and np.sum(np.abs(self.current_position)) > 0:
            ridge_penalty = self.lmd_ridge * np.sum((weights - self.current_position) ** 2)  # L2 penalty
        else:
            ridge_penalty = 0

        return portfolio_variance + ridge_penalty
    

    class EqualRiskContributionStrategy(OptimizationStrategy):
    """
    Stratégie Equal Risk Contribution (ERC), où chaque actif contribue également au risque total.
    """
    def __init__(self, lmd_mu: float = 0.0, lmd_var: float = 0.0, **kwargs) -> None:
        """
        Initialise la stratégie ERC avec des paramètres pour pondérer rendement et variance.

        Args:
            lmd_mu (float): Pondération pour maximiser le rendement.
            lmd_var (float): Pondération pour minimiser la variance.
        """
        super().__init__(**kwargs)
        self.lmd_mu = lmd_mu
        self.lmd_var = lmd_var

    def objective_function(self, weights: np.ndarray, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> float:
        """
        Fonction objectif pour équilibrer la contribution au risque.

        Args:
            weights (np.ndarray): Poids du portefeuille.
            expected_returns (pd.Series): Rendements attendus.
            cov_matrix (pd.DataFrame): Matrice de covariance.

        Returns:
            float: Valeur de la fonction objectif ERC.
        """

        cov_matrix = np.array(cov_matrix)

        risk_contributions = ((cov_matrix @ weights) * weights) / np.sqrt((weights.T @ cov_matrix @ weights))
        risk_objective = np.sum((risk_contributions[:, None] - risk_contributions[None, :])**2)

        return_value_objective = -self.lmd_mu * weights.T @ expected_returns
        variance_objective = self.lmd_var * weights.T @ cov_matrix @ weights

        # Terme de régularisation Ridge (si max_turnover est défini)
        if self.max_turnover is not None and hasattr(self, "current_position") and np.sum(np.abs(self.current_position)) > 0:
            ridge_penalty = self.lmd_ridge * np.sum((weights - self.current_position) ** 2)  # L2 penalty
        else:
            ridge_penalty = 0

        return risk_objective + return_value_objective + variance_objective + ridge_penalty
    
class ValueStrategy(RankedStrategy):
    """
    Stratégie basée sur la valeur relative des actifs (ratio prix actuel / prix passé).
    """
    def rank_assets(self, historical_data: pd.DataFrame) -> pd.Series:
        """
        Classe les actifs par leur ratio de valeur relative.

        Args:
            historical_data (pd.DataFrame): Les données historiques.

        Returns:
            pd.Series: Classement des actifs (meilleure valeur = rang élevé).
        """
        last_prices = historical_data.iloc[-1]  # Dernier prix
        prices_one_year_ago = historical_data.iloc[0]  # Prix il y a un an
        coef_asset = last_prices / prices_one_year_ago
        return coef_asset.rank(ascending=False, method='first')

class MomentumStrategy(RankedStrategy):
    """
    Stratégie Momentum basée sur les performances passées des actifs.
    """
    def rank_assets(self, historical_data: pd.DataFrame) -> pd.Series:
        """
        Classe les actifs par leur performance passée.

        Args:
            historical_data (pd.DataFrame): Les données historiques.

        Returns:
            pd.Series: Classement des actifs (meilleures performances = rang élevé).
        """
        returns = historical_data.pct_change().dropna()
        len_window = len(returns)
        delta = int(np.ceil(len_window * (1 / 12)))
        total_returns = returns.rolling(window=len_window - delta).apply(lambda x: (1 + x).prod() - 1)
        latest_returns = total_returns.iloc[-delta]
        latest_returns = latest_returns.dropna()
        return latest_returns.rank(ascending=True, method='first')

class MinVolStrategy(RankedStrategy):
    def rank_assets(self, historical_data: pd.DataFrame) -> pd.Series:
        """
        Classe les actifs en fonction de leur volatilité, où les actifs moins volatils sont favorisés.

        Args:
            historical_data (pd.DataFrame): Les données historiques.

        Returns:
            pd.Series: Classement des actifs en fonction de la volatilité.
        """
        returns = historical_data.pct_change().dropna()
        volatility = returns.std()
        volatility.dropna()
        return volatility.rank(ascending=False, method='first').sort_values()

class LongOnlyMomentumStrategy(EqualWeightStrategy):
    """
    Stratégie long-only basée sur le momentum à 1 an.

    Cette classe calcule le momentum sur une période donnée (par défaut 252 jours) 
    et sélectionne, selon un quantile défini, les actifs ayant le meilleur momentum.
    Les actifs sélectionnés se voient attribuer des poids égaux.

    Args:
        quantile (float): Le quantile pour la sélection des actifs.
                          Exemples :
                          - 0.1 pour une construction en décile (top 10%),
                          - 0.2 pour une construction en quintile (top 20%),
                          - 0.25 pour une construction en quartile (top 25%).
        groupby_dict (dict, optionnel): Dictionnaire avec en clé les tickers et en valeur 
                          le nom du secteur associé. Si fourni, le quantile sera appliqué 
                          par secteur.
    """
    def __init__(self, quantile: float, groupby_dict: dict = None):
        self.quantile = quantile
        self.groupby_dict = groupby_dict

    def signals(self, historical_data: pd.DataFrame) -> list:
        """
        Calcule la position (poids) en sélectionnant les actifs avec le meilleur momentum 
        selon le quantile défini. Si groupby_dict est fourni, la sélection est faite 
        par secteur (c'est-à-dire que pour chaque secteur, on sélectionne le quantile 
        des meilleurs actifs).
        
        Args:
            historical_data (pd.DataFrame): Données historiques des prix (indexées par date).

        Returns:
            list: Liste des tickers sélectionnés.
        """
        # Calcul du momentum sur l'intégralité de la période
        momentum = historical_data.pct_change(periods=len(historical_data)-1).iloc[-1]
        
        if self.groupby_dict is None:
            # Sélection globale : tri décroissant et sélection du top quantile
            sorted_momentum = momentum.sort_values(ascending=False)
            n_assets = len(sorted_momentum)
            n_selected = int(np.ceil(n_assets * self.quantile))
            if n_selected == 0:
                n_selected = 1  # au moins un actif
            selected_assets = sorted_momentum.iloc[:n_selected]
            return selected_assets.index.tolist()
        else:
            # Création d'un DataFrame avec le momentum et le secteur associé
            momentum_df = momentum.to_frame(name="momentum")
            momentum_df["sector"] = momentum_df.index.map(self.groupby_dict.get)
            # On élimine les actifs dont le secteur n'est pas défini
            momentum_df = momentum_df.dropna(subset=["sector"])
            
            selected_list = []
            # Pour chaque secteur, on trie par momentum décroissant
            # et on sélectionne le top quantile des actifs
            for sector, group in momentum_df.groupby("sector"):
                group_sorted = group.sort_values(by="momentum", ascending=False)
                n_group = len(group_sorted)
                n_selected_group = int(np.ceil(n_group * self.quantile))
                if n_selected_group == 0:
                    n_selected_group = 1  # au moins un actif par secteur
                selected_group = group_sorted.iloc[:n_selected_group]
                selected_list.extend(selected_group.index.tolist())
            return selected_list
        
class LongOnlyIdiosyncraticMomentumStrategy(EqualWeightStrategy):
    
    def __init__(self, quantile: float, benchmark: pd.Series, window_size: int = 252 * 3, delta: int = 21 * 12, groupby_dict: dict = None):
        """
        :param quantile: pour sélectionner un quantile d'actifs (ex. 0.2 pour les 20% les plus attractifs)
        :param benchmark: série de prix du benchmark (indexée par date)
        :param window_size: taille de la fenêtre de rolling en nombre de jours (par défaut 36 mois)
        :param delta: période d'estimation du momentum idiosyncratique (par défaut 12 mois en jours)
        :param groupby_dict: dictionnaire facultatif associant chaque ticker (clé) à un groupe/secteur (valeur)
        """
        self.quantile = quantile
        self.window_size = window_size
        self.delta = delta
        self.groupby_dict = groupby_dict
        
        # Conversion du benchmark en rendements journaliers
        self.benchmark_returns = benchmark.pct_change().dropna()
        # Pré-calcul des rolling mean et variance pour le benchmark
        self.rolling_mean_X = self.benchmark_returns.rolling(window=self.window_size).mean()
        self.rolling_var_X = self.benchmark_returns.rolling(window=self.window_size).var()

    def signals(self, historical_data: pd.DataFrame) -> list:
        """
        historical_data : DataFrame contenant, pour chaque date (index) et pour chaque actif (colonnes),
        les prix journaliers (ou rendements déjà calculés).
        On suppose ici disposer d'au moins 36 mois de données.
        """
        # Calculer les rendements journaliers des actifs
        returns = historical_data.pct_change().dropna()
        # Conserver uniquement les dates communes aux rendements et au benchmark
        common_dates = returns.index.intersection(self.benchmark_returns.index)
        returns = returns.loc[common_dates]
        benchmark_returns = self.benchmark_returns.loc[common_dates]
        
        # Définir la période d'estimation du momentum idiosyncratique : sur les delta derniers jours
        try:
            window_end = returns.index[-1]
        except IndexError:
            return []
        window_start = window_end - pd.Timedelta(days=self.delta)
        
        # Calcul vectorisé des rolling means pour tous les actifs
        rolling_mean_Y_all = returns.rolling(window=self.window_size).mean()
        # Calcul vectorisé de la covariance roulante entre chaque actif et le benchmark
        rolling_cov_all = returns.rolling(window=self.window_size).cov(benchmark_returns)
        
        # Calcul de beta et alpha pour chaque actif et chaque date
        # beta = rolling_cov_all / rolling_var_X, et alpha = rolling_mean_Y_all - beta * rolling_mean_X
        beta_df = rolling_cov_all.divide(self.rolling_var_X, axis=0)
        alpha_df = rolling_mean_Y_all - beta_df.multiply(self.rolling_mean_X, axis=0)
        
        # Calcul des résidus quotidiens : resid = rendement observé - (alpha + beta * rendement benchmark)
        resid_df = returns - (alpha_df + beta_df.multiply(benchmark_returns, axis=0))
        
        # Restreindre les résidus à la période d'intérêt (les delta derniers jours)
        resid_period = resid_df.loc[window_start:window_end]
        
        # Calcul du score idiosyncratique pour chaque actif : moyenne des résidus / écart-type des résidus
        momentum_scores = resid_period.mean() / resid_period.std()
        sorted_scores = momentum_scores.sort_values(ascending=False)
        
        # Si tous les scores sont nuls ou NaN, retourner une liste vide
        if sorted_scores.isnull().all() or sorted_scores.eq(0).all():
            return []
        
        # Si un groupby_dict est fourni, faire la sélection par groupe
        if self.groupby_dict is not None:
            # Créer un DataFrame avec les scores et le groupe/secteur associé
            score_df = sorted_scores.to_frame(name="score")
            score_df["group"] = score_df.index.map(self.groupby_dict.get)
            # Éliminer les tickers sans groupe défini
            score_df = score_df.dropna(subset=["group"])
            selected_list = []
            # Pour chaque groupe, trier par score décroissant et sélectionner le quantile défini
            for group, group_data in score_df.groupby("group"):
                group_sorted = group_data.sort_values("score", ascending=False)
                n_group = len(group_sorted)
                n_selected_group = int(np.ceil(n_group * self.quantile))
                if n_selected_group == 0:
                    n_selected_group = 1
                selected_group = group_sorted.iloc[:n_selected_group]
                selected_list.extend(selected_group.index.tolist())
            return selected_list
        else:
            # Sélection globale : tri décroissant et sélection du top quantile
            n_assets = len(sorted_scores)
            n_selected = int(np.ceil(n_assets * self.quantile))
            if n_selected == 0:
                n_selected = 1
            selected_assets = sorted_scores.iloc[:n_selected]
            return selected_assets.index.tolist()

class LongOnlySharpeStrategy(EqualWeightStrategy):
    """
    Stratégie long-only basée sur le Sharpe ratio.

    Cette classe calcule le Sharpe ratio sur une période donnée à partir des rendements journaliers,
    et sélectionne, selon un quantile défini, les actifs ayant le meilleur Sharpe ratio.
    Les actifs sélectionnés se voient attribuer des poids égaux.

    Args:
        quantile (float): Le quantile pour la sélection des actifs.
                          Exemples :
                          - 0.1 pour une construction en décile (top 10%),
                          - 0.2 pour une construction en quintile (top 20%),
                          - 0.25 pour une construction en quartile (top 25%).
        groupby_dict (dict, optionnel): Dictionnaire avec en clé les tickers et en valeur 
                          le nom du secteur associé. Si fourni, la sélection sera faite 
                          par secteur (i.e. le quantile sera appliqué sur chaque groupe).
    """
    def __init__(self, quantile: float, groupby_dict: dict = None):
        self.quantile = quantile
        self.groupby_dict = groupby_dict

    def signals(self, historical_data: pd.DataFrame) -> list:
        """
        Calcule les positions (poids) en sélectionnant les actifs avec le meilleur Sharpe ratio 
        selon le quantile défini. Si groupby_dict est fourni, la sélection est faite 
        par secteur (c'est-à-dire que pour chaque secteur, on sélectionne le quantile 
        des meilleurs actifs).

        Args:
            historical_data (pd.DataFrame): Données historiques des prix (indexées par date).

        Returns:
            list: Liste des tickers sélectionnés.
        """
        # Calculer les rendements journaliers
        returns = historical_data.pct_change().dropna()
        # Calculer le Sharpe ratio pour chaque actif : moyenne / écart-type
        sharpe = returns.mean() / returns.std()
        
        if self.groupby_dict is None:
            # Sélection globale : trier par Sharpe décroissant et sélectionner le top quantile
            sorted_sharpe = sharpe.sort_values(ascending=False)
            n_assets = len(sorted_sharpe)
            n_selected = int(np.ceil(n_assets * self.quantile))
            if n_selected == 0:
                n_selected = 1  # au moins un actif
            selected_assets = sorted_sharpe.iloc[:n_selected]
            return selected_assets.index.tolist()
        else:
            # Créer un DataFrame avec le Sharpe et le secteur associé
            sharpe_df = sharpe.to_frame(name="sharpe")
            sharpe_df["sector"] = sharpe_df.index.map(self.groupby_dict.get)
            # Éliminer les tickers dont le secteur n'est pas défini
            sharpe_df = sharpe_df.dropna(subset=["sector"])
            
            selected_list = []
            # Pour chaque secteur, trier par Sharpe décroissant et sélectionner le top quantile
            for sector, group in sharpe_df.groupby("sector"):
                group_sorted = group.sort_values(by="sharpe", ascending=False)
                n_group = len(group_sorted)
                n_selected_group = int(np.ceil(n_group * self.quantile))
                if n_selected_group == 0:
                    n_selected_group = 1
                selected_group = group_sorted.iloc[:n_selected_group]
                selected_list.extend(selected_group.index.tolist())
            return selected_list
        
class ValueTactical(AlphaBasedTactical):
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
    
class MomentumTactical(AlphaBasedTactical):
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
    
