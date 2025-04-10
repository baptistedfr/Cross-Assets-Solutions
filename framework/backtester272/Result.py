import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates
import quantstats as qs

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class Result:
    """
    Classe pour stocker et analyser les résultats d'un backtest.

    Cette classe permet de calculer des métriques de performance, visualiser les résultats 
    sous différentes formes (tableaux, graphiques), et comparer plusieurs stratégies.
    """

    def __init__(self, performance: pd.Series, 
                 weight: pd.DataFrame, 
                 total_transactions_cost: float, 
                 name: str = None, 
                 benchmark: pd.Series = None, 
                 benchmark_weight: pd.Series = None,
                 tactical_performance: pd.Series = None,
                 tactical_weight: pd.DataFrame = None,
                 tactical_benchmark: pd.Series = None,
                 tactical_benchmark_weight: pd.DataFrame = None,
                 macro_performance: pd.Series = None,
                 macro_performance_weight: pd.DataFrame = None,
                 macro_benchmark: pd.Series = None,
                 macro_benchmark_weight: pd.DataFrame = None,
                 tactical_macro_performance: pd.Series = None,
                 tactical_macro_performance_weight: pd.DataFrame = None,
                 tactical_macro_benchmark: pd.Series = None,
                 tactical_macro_benchmark_weight: pd.DataFrame = None
                 ):
        """
        Initialise les résultats du backtest.

        Args:
            performance (pd.Series): Série temporelle représentant la performance cumulée du portefeuille.
            weight (pd.DataFrame): Poids des actifs dans le portefeuille au fil du temps.
            total_transactions_cost (float): Coût total des transactions pendant le backtest.
            name (str, optional): Nom de la stratégie associée aux résultats.
            benchmark (pd.Series, optional): Série temporelle représentant la performance du benchmark.
        """
        self.performance = performance
        self.weights = weight
        self.total_transactions_cost = total_transactions_cost
        self.name = name
        self.benchmark = benchmark
        self.benchmark_weight = benchmark_weight
        self.tactical_performance = tactical_performance
        self.tactical_weight = tactical_weight
        self.tactical_benchmark = tactical_benchmark
        self.tactical_benchmark_weight = tactical_benchmark_weight
        self.macro_performance = macro_performance
        self.macro_performance_weight = macro_performance_weight
        self.macro_benchmark = macro_benchmark
        self.macro_benchmark_weight = macro_benchmark_weight
        self.tactical_macro_performance = tactical_macro_performance
        self.tactical_macro_performance_weight = tactical_macro_performance_weight
        self.tactical_macro_benchmark = tactical_macro_benchmark
        self.tactical_macro_benchmark_weight = tactical_macro_benchmark_weight

    def periods_freq(self, series: pd.Series) -> int:
        """
        Calcule la fréquence des données en jours (252 pour données boursières, sinon 365).

        Args:
            series (pd.Series): Série temporelle à analyser.

        Returns:
            int: Fréquence estimée (252 ou 365).
        """
        serie_length = len(series)
        num_of_days = (series.index[-1] - series.index[0]).days
        ratio = serie_length / num_of_days
        
        if abs(ratio - 1) < abs(ratio - (252 / 365)):
            return 365
        else:
            return 252
        
    def annualized_transactions_cost(self) -> float:
        """
        Calcule le coût annuel moyen des transactions.

        Returns:
            float: Coût annuel moyen des transactions.
        """
        return self.total_transactions_cost * self.periods_freq(self.performance) / len(self.performance) / 100

    def volatility(self, prices: pd.Series) -> float:
        """
        Calcule la volatilité annualisée d'une série de prix.

        Args:
            prices (pd.Series): Série temporelle des prix.

        Returns:
            float: Volatilité annualisée.
        """
        returns = prices.pct_change().dropna()
        return returns.std() * (self.periods_freq(prices) ** 0.5)

    def downside_volatility(self, prices: pd.Series) -> float:
        """
        Calcule la volatilité de downside annualisée d'une série de prix,
        en considérant uniquement les rendements en dessous du MAR (Minimum Acceptable Return).

        Args:
            prices (pd.Series): Série temporelle des prix.

        Returns:
            float: Volatilité de downside annualisée.
        """
        returns = prices.pct_change().dropna()
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        downside_std = downside_returns.std()
        return downside_std * (self.periods_freq(prices) ** 0.5)


    def perf(self, prices: pd.Series) -> float:
        """
        Calcule la performance totale d'une série de prix.

        Args:
            prices (pd.Series): Série temporelle des prix.

        Returns:
            float: Performance totale (en pourcentage).
        """
        return prices[-1] / prices[0] - 1

    def cagr(self, prices: pd.Series) -> float:
        """
        Calcule le taux de croissance annuel composé (CAGR).

        Args:
            prices (pd.Series): Série temporelle des prix.

        Returns:
            float: CAGR (en pourcentage).
        """
        total_periods = len(prices)
        total_years = total_periods / self.periods_freq(prices)
        return (self.perf(prices) + 1) ** (1 / total_years) - 1

    def max_drawdown(self, prices: pd.Series) -> float:
        """
        Calcule le drawdown maximal d'une série de prix.

        Args:
            prices (pd.Series): Série temporelle des prix.

        Returns:
            float: Drawdown maximal (en pourcentage).
        """
        drawdown = (prices / prices.cummax() - 1)
        return drawdown.min()

    def sharpe_ratio(self, prices: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calcule le ratio de Sharpe d'une série de prix.

        Args:
            prices (pd.Series): Série temporelle des prix.
            risk_free_rate (float): Taux sans risque (par défaut 0).

        Returns:
            float: Ratio de Sharpe.
        """
        annualised_return = self.cagr(prices)
        annualised_volatility = self.volatility(prices)
        return (annualised_return - risk_free_rate) / annualised_volatility if annualised_volatility != 0 else np.nan
        # returns = prices.pct_change().dropna()
        # excess_returns = returns - risk_free_rate / self.periods_freq(prices)
        # return excess_returns.mean() / excess_returns.std() * (self.periods_freq(prices) ** 0.5)
    
    def sortino_ratio(self, prices: pd.Series, risk_free_rate: float = 0.0, mar: float = 0.0) -> float:
        """
        Calcule le ratio de Sortino d'une série de prix.

        Args:
            prices (pd.Series): Série temporelle des prix.
            risk_free_rate (float): Taux sans risque annualisé (par défaut 0).
            mar (float): Minimum Acceptable Return (par défaut 0).

        Returns:
            float: Ratio de Sortino annualisé.
        """
        downside_volatility = self.downside_volatility(prices)
        annualised_return = self.cagr(prices)
        return (annualised_return - risk_free_rate) / downside_volatility if downside_volatility != 0 else np.nan

    def value_at_risk(self, prices: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calcule la Value at Risk (VaR) historique d'une série de prix.

        Args:
            prices (pd.Series): Série temporelle des prix.
            confidence_level (float): Niveau de confiance (par défaut 95%).

        Returns:
            float: VaR au niveau de confiance donné (rendement négatif).
        """
        returns = prices.pct_change().dropna()
        return np.percentile(returns, (1 - confidence_level) * 100)

    def conditional_value_at_risk(self, prices: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calcule la Conditional Value at Risk (CVaR) historique (aussi appelée Expected Shortfall).

        Args:
            prices (pd.Series): Série temporelle des prix.
            confidence_level (float): Niveau de confiance (par défaut 95%).

        Returns:
            float: CVaR (perte moyenne au-delà de la VaR).
        """
        returns = prices.pct_change().dropna()
        var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
        tail_losses = returns[returns <= var_threshold]
        
        if tail_losses.empty:
            return 0.0  # Pas de pertes extrêmes observées
        return tail_losses.mean()

    def compute_hit_ratio(self, performance: pd.Series, weights: pd.DataFrame):
        """
        Calcule le hit ratio, l'average win et l'average loss à partir de la série de performance
        (prix en base 100) et du DataFrame de weights dont l'index contient les dates de rebalancement.
        
        Pour chaque période allant de la date i à la date i+1 (issues de l'intersection des dates de weights
        et de performance), le rendement est calculé. Si ce rendement est positif, il est ajouté à wins, sinon à loses.
        
        Args:
            performance (pd.Series): Série avec en index la date et en valeur le prix en base 100.
            weights (pd.DataFrame): DataFrame avec en index les dates de rebalancement, en colonnes les tickers,
                                    et en data les poids.
        
        Returns:
            tuple: (hit_ratio, average_win, average_loss)
                hit_ratio = nombre de gains / nombre de pertes (0 si aucune perte),
                average_win = moyenne des rendements positifs (0 si aucun gain),
                average_loss = moyenne des rendements négatifs (0 si aucune perte).
        """

        if weights is None:
            return 0, 0, 0
        
        # Récupérer les dates communes entre weights et performance
        common_dates = weights.index.intersection(performance.index)
        common_dates = common_dates.sort_values()
        
        # Si on n'a pas au moins 2 dates pour calculer un rendement, on renvoie 0 pour tous les metrics
        if len(common_dates) < 2:
            return 0, 0, 0
        
        wins = []
        loses = []
        
        # Calculer le rendement entre chaque paire de dates successives
        for i in range(len(common_dates) - 1):
            date_current = common_dates[i]
            date_next = common_dates[i + 1]
            # Calculer le rendement sous forme de pourcentage
            ret = (performance.loc[date_next] / performance.loc[date_current]) - 1
            if ret > 0:
                wins.append(ret)
            elif ret < 0:
                loses.append(ret)
            # On ignore le cas ret == 0
            
        hit_ratio = (len(wins) / (len(loses) + len(wins))) if (len(loses) + len(wins)) > 0 else 0
        average_win = np.mean(wins) if wins else 0
        average_loss = np.mean(loses) if loses else 0
        
        return hit_ratio, average_win, average_loss

    def get_metrics(self, performance: pd.Series = None, benchmark: pd.Series = None, weights: pd.DataFrame = None) -> dict:
        """
        Calcule et retourne un dictionnaire des principales métriques de performance.

        Returns:
            dict: Dictionnaire des métriques (Performance, CAGR, Volatilité, Drawdown, Sharpe Ratio).
        """
        if performance is None:
            performance = self.performance
            annualized_transactions_cost = self.annualized_transactions_cost()
        else:
            annualized_transactions_cost = 0

        win_rate = qs.stats.win_rate(performance, aggregate="month")
        average_win, average_loss = qs.stats.avg_win(performance, aggregate="month"), qs.stats.avg_loss(performance, aggregate="month")

        metrics = {
            'Performance': f"{self.perf(performance):.2%}",
            'CAGR': f"{self.cagr(performance):.2%}",
            'Volatility': f"{self.volatility(performance):.2%}",
            'Downside Volatility': f"{self.downside_volatility(performance):.2%}",
            'Max Drawdown': f"{self.max_drawdown(performance):.2%}",
            'Max Drawdown Date': self.max_drawdown_date(performance).strftime('%Y-%m-%d'),
            'Sharpe Ratio': f"{self.sharpe_ratio(performance):.2f}",
            'Sortino Ratio': f"{self.sortino_ratio(performance):.2f}",
            'Calmar Ratio': f"{self.calmar_ratio(performance):.2f}",
            'VaR (95%)': f"{self.value_at_risk(performance):.2%}",
            'CVaR (95%)': f"{self.conditional_value_at_risk(performance):.2%}",
            # 'Win Rate': f"{win_rate:.2%}",
            # 'Average Win': f"{average_win:.2%}",
            # 'Average Loss': f"{average_loss:.2%}",
        }

        if benchmark is not None:
            metrics['Tracking Error'] = f"{self.calculate_tracking_error(performance, benchmark):.2%}"
            metrics['Beta'] = f"{self.beta(performance, benchmark):.2f}"
            metrics['Treynor Ratio'] = f"{self.treynor_ratio(performance, benchmark):.2f}"
            metrics['Alpha'] = f"{self.alpha(performance, benchmark):.2%}"
            metrics['Information Ratio'] = f"{self.information_ratio(performance, benchmark):.2f}"

        return metrics

    def show_metrics(self) -> None:
        """
        Affiche les métriques de performance dans un format lisible.
        """
        metrics = self.get_metrics()
        print(pd.Series(metrics))

    def calculate_drawdown(self, performance: pd.Series = None) -> pd.Series:
        """
        Calcule le drawdown à chaque point dans une série temporelle.

        Returns:
            pd.Series: Série des drawdowns.
        """
        if performance is None:
            performance = self.performance
        return performance / performance.cummax() - 1
    
    def max_drawdown_date(self, performance: pd.Series) -> pd.Timestamp:
        """Date du drawdown maximal."""
        drawdowns = self.calculate_drawdown(performance)
        return drawdowns.idxmin()
    
    def calmar_ratio(self, performance) -> float:
        """Calcul du ratio de Calmar."""
        ann_return = self.cagr(performance)
        max_dd = self.max_drawdown(performance)
        return ann_return / abs(max_dd) if max_dd !=0 else np.nan

    def calculate_tracking_error(self, performance: pd.Series = None, benchmark: pd.Series = None) -> float:
        """
        Calcule l'erreur de suivi (tracking error) par rapport à un benchmark.

        Args:
            benchmark (pd.Series): Série temporelle du benchmark.

        Returns:
            float: Erreur de suivi (tracking error).
        """
        if performance is None:
            performance = self.performance
        if benchmark is None:
            benchmark = self.benchmark

        return (performance.pct_change() - benchmark.pct_change()).dropna().std() * (self.periods_freq(performance) ** 0.5)
    
    def beta(self, performance: pd.Series = None, benchmark: pd.Series = None) -> float:
        """Calcul du beta par rapport au benchmark."""
        returns = performance.pct_change().dropna()
        benchmark_returns = benchmark.pct_change().dropna()
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = benchmark_returns.var()
        return covariance / benchmark_variance
    
    def treynor_ratio(self, performance: pd.Series = None, benchmark: pd.Series = None, risk_free_rate: float = 0.0) -> float:
        """Calcul du ratio de Treynor."""
        ann_return = self.cagr(performance)
        beta = self.beta(performance, benchmark)
        return (ann_return - risk_free_rate) / beta
    
    def alpha(self, performance: pd.Series = None, benchmark: pd.Series = None, risk_free_rate: float = 0.0) -> float:
        """Calcul de l'alpha du portefeuille"""
        ann_ptf_return = self.cagr(performance)
        beta = self.beta(performance, benchmark)
        ann_market_return = self.cagr(benchmark)

        return ann_ptf_return - risk_free_rate - beta*(ann_market_return-risk_free_rate)
    
    def information_ratio(self, performance: pd.Series = None, benchmark: pd.Series = None, risk_free_rate: float = 0.0) -> float:
        """Calcul du ratio d'information."""
        alpha = self.alpha(performance, benchmark, risk_free_rate)
        tracking_error = self.calculate_tracking_error(performance, benchmark)
        return alpha/tracking_error

    def compare(self, highlighted_strat: str, *other_results: 'Result') -> None:
        """
        Compare les résultats de plusieurs stratégies avec des graphiques et un tableau de métriques.

        Args:
            *other_results (Result): Autres résultats de backtest à comparer.
        """
        results = [self] + list(other_results)

        # Préparation des données

        # On ajoute d'abord les benchmarks classiques, tactiques, macro et tactiques macro s'ils existent.
        performances = (
            ([self.benchmark] if self.benchmark is not None else [])
            + ([self.tactical_benchmark] if self.tactical_benchmark is not None else [])
            + ([self.macro_benchmark] if self.macro_benchmark is not None else [])
            + ([self.tactical_macro_benchmark] if self.tactical_macro_benchmark is not None else [])
            # Ensuite, pour chaque résultat, on ajoute les performances classiques,
            # tactiques, macro et tactiques macro s'ils sont disponibles.
            + [result.performance for result in results]
            + [result.tactical_performance for result in results if result.tactical_performance is not None]
            + [result.macro_performance for result in results if hasattr(result, 'macro_performance') and result.macro_performance is not None]
            + [result.tactical_macro_performance for result in results if hasattr(result, 'tactical_macro_performance') and result.tactical_macro_performance is not None]
        )
        
        metrics = (
            ([self.get_metrics(performance=self.benchmark, weights=self.benchmark_weight)] if self.benchmark is not None else [])
            + ([self.get_metrics(performance=self.tactical_benchmark, benchmark=self.benchmark, weights=self.tactical_benchmark_weight)] if self.tactical_benchmark is not None else [])
            + ([self.get_metrics(performance=self.macro_benchmark, benchmark=self.benchmark, weights=self.macro_benchmark_weight)] if self.macro_benchmark is not None else [])
            + ([self.get_metrics(performance=self.tactical_macro_benchmark, benchmark=self.benchmark, weights=self.tactical_macro_benchmark_weight)] if self.tactical_macro_benchmark is not None else [])
            + [result.get_metrics(benchmark=self.benchmark, weights=result.weights) for result in results]
            + [result.get_metrics(performance=result.tactical_performance, benchmark=self.benchmark, weights=result.tactical_weight) for result in results if result.tactical_performance is not None]
            + [result.get_metrics(performance=self.macro_performance, benchmark=self.benchmark, weights=result.macro_performance_weight) for result in results if hasattr(result, 'macro_benchmark') and result.macro_benchmark is not None]
            + [result.get_metrics(performance=result.tactical_macro_performance, benchmark=self.benchmark, weights=result.tactical_macro_performance_weight) for result in results if hasattr(result, 'tactical_macro_benchmark') and result.tactical_macro_benchmark is not None]
        )
        
        drawdowns = (
            ([self.calculate_drawdown(self.benchmark)] if self.benchmark is not None else [])
            + ([self.calculate_drawdown(self.tactical_benchmark)] if self.tactical_benchmark is not None else [])
            + ([self.calculate_drawdown(self.macro_benchmark)] if self.macro_benchmark is not None else [])
            + ([self.calculate_drawdown(self.tactical_macro_benchmark)] if self.tactical_macro_benchmark is not None else [])
            + [result.calculate_drawdown() for result in results]
            + [result.calculate_drawdown(result.tactical_performance) for result in results if result.tactical_performance is not None]
            + [result.calculate_drawdown(result.macro_performance) for result in results if hasattr(result, 'macro_performance') and result.macro_performance is not None]
            + [result.calculate_drawdown(result.tactical_macro_performance) for result in results if hasattr(result, 'tactical_macro_performance') and result.tactical_macro_performance is not None]
        )

        returns = (
            ([self.benchmark.pct_change().dropna()] if self.benchmark is not None else [])
            + ([self.tactical_benchmark.pct_change().dropna()] if self.tactical_benchmark is not None else [])
            + ([self.macro_benchmark.pct_change().dropna()] if self.macro_benchmark is not None else [])
            + ([self.tactical_macro_benchmark.pct_change().dropna()] if self.tactical_macro_benchmark is not None else [])
            + [result.performance.pct_change().dropna() for result in results]
            + [result.tactical_performance.pct_change().dropna() for result in results if result.tactical_performance is not None]
            + [result.macro_performance.pct_change().dropna() for result in results if hasattr(result, 'macro_performance') and result.macro_performance is not None]
            + [result.tactical_macro_performance.pct_change().dropna() for result in results if hasattr(result, 'tactical_macro_performance') and result.tactical_macro_performance is not None]
        )

        names = (
            (['Benchmark'] if self.benchmark is not None else [])
            + (['Tactical Benchmark'] if self.tactical_benchmark is not None else [])
            + (['Macro Benchmark'] if self.macro_benchmark is not None else [])
            + (['Tactical Macro Benchmark'] if self.tactical_macro_benchmark is not None else [])
            + [result.name for result in results]
            + [result.name + " Tactical" for result in results if result.tactical_performance is not None]
            + [result.name + " Macro" for result in results if hasattr(result, 'macro_performance') and result.macro_performance is not None]
            + [result.name + " Tactical Macro" for result in results if hasattr(result, 'tactical_macro_performance') and result.tactical_macro_performance is not None]
        )

        if self.benchmark is not None:
            # Rajoute la métrique de tracking error pour le benchmark classique
            metrics[0]['Tracking Error'] = self.calculate_tracking_error(self.benchmark)
            metrics[0]['Beta'] = "0.00"
            metrics[0]['Treynor Ratio'] = "0.00"
            metrics[0]['Alpha'] = "0.00%"
            metrics[0]['Information Ratio'] = "0.00"

        # Calcul des rendements annuels (EOY Returns)
        eoy_returns = []
        for perf in performances:
            annual_returns = perf.resample('YE').last().pct_change(fill_method=None).dropna()
            eoy_returns.append(annual_returns)

        # Création de la figure avec GridSpec
        num_results = len(names)
        height = 8
        height += 2 if self.benchmark is not None else 0
        height += 2 if self.tactical_performance is not None or self.macro_performance is not None else 0
        width = 12 + len(names)*2

        increment = 0

        fig = plt.figure(figsize=(width, height * 3))  # Ajustement de la largeur
        
        gs = fig.add_gridspec(height, max(1, num_results), hspace=0.6, wspace=0.03)  # hspace ajusté, wspace réduit

        # Performance (prend deux lignes)
        ax_perf = fig.add_subplot(gs[increment:increment+2, :])
        sns.set(style="whitegrid")
        for perf, name in zip(performances[::-1], names[::-1]):
            if name in highlighted_strat:
                ax_perf.plot(perf.index, perf, label=name, linewidth=2)
            else :
                if name == 'Benchmark':
                    ax_perf.plot(perf.index, perf, label=name, color='black', linewidth=2, linestyle='--', alpha=0.5)
                else:
                    # ax_perf.plot(perf.index, perf, label=name, linewidth=2, alpha=0.2)
                    pass
                
        ax_perf.set_title("Performance des stratégies", fontsize=16)
        ax_perf.set_ylabel("Valeur")
        ax_perf.legend(loc="upper left", fontsize=10)
        # Ligne de base à 0 en pointillés
        ax_perf.axhline(perf.iloc[0], color='black', linestyle='--', linewidth=1)
        ax_perf.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{(x - perf.iloc[0])/perf.iloc[0]:.0%}'))
        ax_perf.grid(True)
        increment += 2

        # Performance (échelle log)
        ax_log = fig.add_subplot(gs[increment:increment+2, :])
        sns.set(style="whitegrid")
        for perf, name in zip(performances[::-1], names[::-1]):
            discrete_index = np.arange(len(perf))  # Convert index to discrete time steps
            if name == 'Benchmark':
                ax_log.plot(discrete_index, np.log(perf), label=name, color='black', linewidth=2, linestyle='--')
            else:
                ax_log.plot(discrete_index, np.log(perf), label=name, linewidth=2)

        ax_log.set_title("Performance des stratégies", fontsize=16)
        ax_log.set_ylabel("Valeur")
        ax_log.set_yscale("log")  # Passage en échelle logarithmique
        ax_log.legend(loc="upper left", fontsize=10)
        # Ligne de base à la valeur initiale en pointillés
        ax_log.axhline(np.log(perf.iloc[0]), color='black', linestyle='--', linewidth=1)
        # Format de l'échelle pour afficher le rendement par rapport à la valeur initiale
        ax_log.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{(np.exp(x)/perf.iloc[0]-1):.0%}'))
        ax_log.grid(True, which="both")
        increment += 2


        if self.benchmark is not None:
            # Tracking Error relative to benchmark
            ax_te = fig.add_subplot(gs[increment:increment+2, :])
            for perf, name in zip(performances[1:], names[1:]):
                te = (perf.pct_change() - performances[0].pct_change()).dropna()
                te = (1 + te).cumprod() - 1
                if name in highlighted_strat:
                    ax_te.plot(te.index, te, label=name, linewidth=2)
                else:
                    pass
                    # ax_te.plot(te.index, te, label=name, linewidth=2, alpha=0.2)
            ax_te.set_title("Performance rapport au Benchmark", fontsize=16)
            ax_te.set_ylabel("Écart de performance")
            ax_te.legend(loc="upper left", fontsize=10)
            ax_te.axhline(0, color='black', linestyle='--', linewidth=1)
            ax_te.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
            ax_te.grid(True)
            increment += 2



        if self.tactical_performance is not None or self.macro_performance is not None:
            # Tracking Error relative to strategy
            ax_te = fig.add_subplot(gs[increment:increment+2, :])
            te = (self.tactical_performance.pct_change() - self.performance.pct_change()).dropna()
            te = (1 + te).cumprod() - 1
            ax_te.plot(te.index, te, label="Tactical", linewidth=2)
            te = (self.macro_performance.pct_change() - self.performance.pct_change()).dropna()
            te = (1 + te).cumprod() - 1
            ax_te.plot(te.index, te, label="Macro", linewidth=2)
            te = (self.tactical_macro_performance.pct_change() - self.performance.pct_change()).dropna()
            te = (1 + te).cumprod() - 1
            ax_te.plot(te.index, te, label="Tactical Macro", linewidth=2)
            ax_te.set_title("Performance des tactiques ou dynamiques autour de la stratégie", fontsize=16)
            ax_te.set_ylabel("Écart de performance")
            ax_te.legend(loc="upper left", fontsize=10)
            ax_te.axhline(0, color='black', linestyle='--', linewidth=1)
            ax_te.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
            ax_te.grid(True)
            increment += 2


        # Rendements annuels (EOY Returns)
        ax_eoy = fig.add_subplot(gs[increment, :])
        bar_width = 0.8 / num_results  # Largeur des barres pour chaque stratégie
        for i, (eoy, name) in enumerate(zip(eoy_returns, names)):
            positions = np.arange(len(eoy)) + i * bar_width
            ax_eoy.bar(positions, eoy.values, width=bar_width, label=name)
        ax_eoy.set_title("Rendement annuel (EOY Returns)", fontsize=14)
        ax_eoy.set_ylabel("Rendement (%)")
        ax_eoy.set_xticks(np.arange(len(eoy_returns[0])) + bar_width * (num_results - 1) / 2)
        ax_eoy.set_xticklabels([date.year for date in eoy_returns[0].index], rotation=45)
        ax_eoy.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        ax_eoy.legend(loc="upper left", fontsize=10)
        ax_eoy.grid(True)
        increment += 1

        # Drawdowns
        drawdown_min = min(dd.min() for dd in drawdowns)
        drawdown_max = max(dd.max() for dd in drawdowns)
        date_min = min(dd.index.min() for dd in drawdowns)
        date_max = max(dd.index.max() for dd in drawdowns)
        for i, (dd, name) in enumerate(zip(drawdowns, names)):
            ax_dd = fig.add_subplot(gs[increment, i])
            sns.lineplot(ax=ax_dd, x=dd.index, y=dd, color='red')
            ax_dd.fill_between(dd.index, dd, 0, color='red', alpha=0.3)
            ax_dd.set_title(name, fontsize=14, fontweight='bold')
            if i == 0:
                ax_dd.set_ylabel("Drawdown")
                ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
            else:
                ax_dd.set_ylabel("")
                ax_dd.yaxis.set_ticklabels([])  # Supprimer les labels y
            ax_dd.set_ylim(drawdown_min, drawdown_max)
            ax_dd.set_xlim(date_min, date_max)
            ax_dd.axhline(0, color='black', linestyle='--', linewidth=1)
            ax_dd.grid(True)
            # Suppression des labels x
            ax_dd.set_xlabel("")
            # Réduction de la densité des dates
            ax_dd.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
            ax_dd.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        increment += 1


        # Histogrammes
        ret_min_x = min(r.min() for r in returns)
        ret_max_x = max(r.max() for r in returns)

        # Calculer les limites en y pour les histogrammes
        hist_data = []
        for r in returns:
            hist, _ = np.histogram(r, bins=30, range=(ret_min_x, ret_max_x))
            hist_data.append(hist)

        ret_min_y = 0
        ret_max_y = max(h.max() for h in hist_data)

        for i, (r, name) in enumerate(zip(returns, names)):
            ax_ret = fig.add_subplot(gs[increment, i])
            sns.histplot(ax=ax_ret, data=r, kde=True, bins=30, color='blue')
            if i == 0:
                ax_ret.set_ylabel("Rendements")
            else:
                ax_ret.set_ylabel("")
                ax_ret.yaxis.set_ticklabels([])  # Supprimer les labels y
            ax_ret.set_xlim(ret_min_x, ret_max_x)
            ax_ret.set_ylim(ret_min_y, ret_max_y)
            ax_ret.set_xlabel("Rendements (%)")
            ax_ret.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
            ax_ret.grid(True)

        plt.show()

                
        # Création d'un DataFrame pour les métriques
        metrics_df = pd.DataFrame(metrics)
        metrics_df.index = names
        # metrics_df.loc[self.name] = metrics[0]

        # Mettre en forme le tableau avec l'objet courant en gras
        print(metrics_df.T.to_markdown())


    def visualize(self, highlighted_strat: str) -> None:
        """
        Compare les performances de la stratégie actuelle avec d'autres si disponible.
        """
        self.compare(highlighted_strat)

    def positions(self, status: str = None) -> None:
        """
        Visualise les positions dans un graphique en aires empilées avec Plotly.
        """
        
        # Filtrer les colonnes pertinentes (éliminer celles avec uniquement des 0)
        if status == "benchmark":
            weights = self.benchmark_weight
        elif status == "tactical":
            weights = self.tactical_weight
        elif status == "tactical_benchmark":
            weights = self.tactical_benchmark_weight
        elif status == "macro":
            weights = self.macro_performance_weight
        elif status == "macro_benchmark":
            weights = self.macro_benchmark_weight
        elif status == "tactical_macro":
            weights = self.tactical_macro_performance_weight
        elif status == "tactical_macro_benchmark":
            weights = self.tactical_macro_benchmark_weight
        else:
            weights = self.weights

        weights_filtered = weights.loc[:, (weights != 0).any(axis=0)]

        # Créer une figure Plotly
        fig = go.Figure()

        # Ajouter chaque colonne comme une trace dans le graphique en aires empilées
        for column in weights_filtered.columns:
            y_values = weights_filtered[column] * 100  # Conversion en %

            # Créer une série de textes conditionnels pour le hover
            hover_text = y_values.apply(
                lambda x: f"<b>{column}</b><br>Poids: {x:.2f}%" if x != 0 else ""
            )

            fig.add_trace(go.Scatter(
                x=weights_filtered.index,
                y=y_values,
                mode='lines',
                name=column,
                stackgroup='one',  # Utilisation de stackgroup pour empiler les aires
                text=hover_text,    # Assignation du texte conditionnel
                hovertemplate='%{text}<extra></extra>'  # Utilisation du texte conditionnel dans le hover
            ))

        # Personnalisation de la figure
        fig.update_layout(
            title=f"Évolution des positions - {self.name if status is None else status.capitalize()}",
            xaxis_title="Date",
            yaxis_title="Poids (%)",
            yaxis=dict(ticksuffix="%", showgrid=True),  # Affiche l'ordonnée en %
            legend_title="Positions",
            hovermode="x unified",  # Affiche toutes les positions pour une date donnée
            template="plotly_white",
            width=1000,  # Ajustement de la largeur
            height=600   # Ajustement de la hauteur
        )

        # Afficher la figure
        fig.show()

    def gather_metrics(self, *other_results, highlight_extremes=False) -> None:
        """
        Compare les résultats de plusieurs stratégies avec des graphiques et un tableau de métriques.

        Args:
            *other_results (Result): Autres résultats de backtest à comparer.
        """
        results = [self] + list(other_results)

        metrics = (
            ([self.get_metrics(performance=self.benchmark, weights=self.benchmark_weight)] if self.benchmark is not None else [])
            + ([self.get_metrics(performance=self.tactical_benchmark, benchmark=self.benchmark, weights=self.tactical_benchmark_weight)] if self.tactical_benchmark is not None else [])
            + ([self.get_metrics(performance=self.macro_benchmark, benchmark=self.benchmark, weights=self.macro_benchmark_weight)] if self.macro_benchmark is not None else [])
            + ([self.get_metrics(performance=self.tactical_macro_benchmark, benchmark=self.benchmark, weights=self.tactical_macro_benchmark_weight)] if self.tactical_macro_benchmark is not None else [])
            + [result.get_metrics(benchmark=self.benchmark, weights=result.weights) for result in results]
            + [result.get_metrics(performance=result.tactical_performance, benchmark=self.benchmark, weights=result.tactical_weight) for result in results if result.tactical_performance is not None]
            + [result.get_metrics(benchmark=self.macro_benchmark, weights=result.macro_performance_weight) for result in results if hasattr(result, 'macro_benchmark') and result.macro_benchmark is not None]
            + [result.get_metrics(performance=result.tactical_macro_performance, benchmark=self.benchmark, weights=result.tactical_macro_performance_weight) for result in results if hasattr(result, 'tactical_macro_benchmark') and result.tactical_macro_benchmark is not None]
        )
        
        names = (
            (['Benchmark'] if self.benchmark is not None else [])
            + (['Tactical Benchmark'] if self.tactical_benchmark is not None else [])
            + (['Macro Benchmark'] if self.macro_benchmark is not None else [])
            + (['Tactical Macro Benchmark'] if self.tactical_macro_benchmark is not None else [])
            + [result.name for result in results]
            + [result.name + " Tactical" for result in results if result.tactical_performance is not None]
            + [result.name + " Macro" for result in results if hasattr(result, 'macro_performance') and result.macro_performance is not None]
            + [result.name + " Tactical Macro" for result in results if hasattr(result, 'tactical_macro_performance') and result.tactical_macro_performance is not None]
        )

        if self.benchmark is not None:
            # Rajoute la métrique de tracking error pour le benchmark classique
            metrics[0]['Tracking Error'] = "0.00%"
            metrics[0]['Beta'] = "0.00"
            metrics[0]['Treynor Ratio'] = "0.00"
            metrics[0]['Alpha'] = "0.00%"
            metrics[0]['Information Ratio'] = "0.00"
                
        # Création d'un DataFrame pour les métriques
        metrics_df = pd.DataFrame(metrics)
        
        metrics_df.index = names
        metrics_df.to_excel("Test.xlsx")
        
        if highlight_extremes:
            def highlight_extremes_higher_better(s):
                s = s.str.rstrip('%').astype(float)
                is_max = s == s.max()
                is_min = s == s.min()
                return ['background-color: green' if v else 'background-color: red' if w else '' for v, w in zip(is_max, is_min)]

            def highlight_extremes_lower_better(s):
                s = s.str.rstrip('%').astype(float)
                is_max = s == s.max()
                is_min = s == s.min()
                return ['background-color: red' if v else 'background-color: green' if w else '' for v, w in zip(is_max, is_min)]

            # Appliquer les deux styles en une seule chaîne
            styled_metrics_df = (metrics_df.style
                .apply(highlight_extremes_higher_better, subset=['Performance','CAGR','Max Drawdown','Sharpe Ratio','Sortino Ratio', 'Calmar Ratio', 
                                                                 'VaR (95%)','CVaR (95%)',
                                                                 'Alpha', 'Information Ratio', 'Treynor Ratio'],axis=0)
                                                                  #'Win Rate', 'Average Win', 'Average Loss'], axis=0)
                .apply(highlight_extremes_lower_better, subset=['Volatility', 'Downside Volatility'], axis=0)
                )       

            return styled_metrics_df
        else:
            # Affichage du DataFrame sans mise en forme
            return metrics_df
