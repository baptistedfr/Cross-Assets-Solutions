import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import plotly.graph_objects as go


class MIDASModel:
    """
    Modèle MIDAS pour une variable macro.

    Modèle:
        y_t = α + β * Σ_{j=0}^{K} w(j, θ) * x_{t-j} + ε_t,

    où:
        w(j, θ) = exp(θ1 * j + θ2 * j²) / Σ_{j=0}^{K} exp(θ1 * j + θ2 * j²)
    """

    def __init__(self, y: pd.Series, x: pd.Series, K: int):
        self.y = y.copy()
        self.x = x.copy()
        self.K = K
        self.params = None
        self.weights = None

        # Aligner les séries sur l'index commun
        common_index = self.y.index.intersection(self.x.index)
        self.y = self.y.loc[common_index]
        self.x = self.x.loc[common_index]

        # Récupération des noms pour les plots
        self.sector_name = self.y.name if self.y.name is not None else "Variable dépendante"
        self.macro_name = self.x.name if self.x.name is not None else "Variable explicative"

    def almon_weights(self, theta1, theta2, K=None):
        if K is None:
            K = self.K
        j = np.arange(K + 1)
        w = np.exp(theta1 * j + theta2 * j ** 2)
        return w / np.sum(w)

    def midas_prediction(self, params):
        alpha, beta, theta1, theta2 = params
        w = self.almon_weights(theta1, theta2)
        n = len(self.x)
        preds = []
        for t in range(self.K, n):
            x_lags = self.x.iloc[t - self.K:t + 1].values[::-1]
            preds.append(alpha + beta * np.dot(w, x_lags))
        return np.array(preds)

    def objective(self, params):
        y_obs = self.y.iloc[self.K:].values
        pred = self.midas_prediction(params)
        return pred - y_obs

    def fit(self, initial_guess=None):
        if initial_guess is None:
            initial_guess = [0, 0.1, 0, 0]
        result = least_squares(self.objective, initial_guess)
        self.params = {
            'alpha': result.x[0],
            'beta': result.x[1],
            'theta1': result.x[2],
            'theta2': result.x[3]
        }
        self.weights = self.almon_weights(result.x[2], result.x[3])
        return self.params

    def predict(self):
        if self.params is None:
            raise Exception("Le modèle n'a pas encore été ajusté. Utilisez la méthode fit() d'abord.")
        param_vector = [self.params['alpha'], self.params['beta'],
                        self.params['theta1'], self.params['theta2']]
        return self.midas_prediction(param_vector)

    def plot_fit(self):
        if self.params is None:
            raise Exception("Le modèle n'a pas encore été ajusté. Utilisez la méthode fit() d'abord.")
        y_obs = self.y.iloc[self.K:]
        y_pred = self.predict()

        # Création de la figure avec deux axes y
        fig = go.Figure()
        # Tracé de la série observée (rendement du secteur)
        fig.add_trace(go.Scatter(x=y_obs.index, y=y_obs.values,
                                 mode='lines', name='Observé (Secteur)'))
        # Tracé de la série prédite
        fig.add_trace(go.Scatter(x=y_obs.index, y=y_pred,
                                 mode='lines', name='Prédit (Secteur)'))
        # Tracé de l'évolution de la variable macro (sur un axe secondaire)
        fig.add_trace(go.Scatter(x=self.x.index, y=self.x.values,
                                 mode='lines', name=f"{self.macro_name} (Macro)",
                                 yaxis="y2"))

        # Mise en forme du layout avec un axe secondaire
        fig.update_layout(
            title=f"Ajustement MIDAS<br>Secteur: {self.sector_name} | Macro: {self.macro_name}",
            xaxis_title="Date",
            yaxis=dict(
                title="Rendement du secteur",
                showgrid=False,
                zeroline=False
            ),
            yaxis2=dict(
                title=self.macro_name,
                overlaying='y',
                side='right'
            ),
            hovermode="x"
        )
        fig.show()

class MultiMIDASModel:
    """
    Modèle MIDAS généralisé pour plusieurs variables macro.

    Le modèle est défini par :

        y_t = α + Σ_{i=1}^{M} β_i * Σ_{j=0}^{K_i} w_i(j, θ_i) * x_{i,t-j} + ε_t,

    où, pour chaque variable macro i, la fonction de pondération est :

        w_i(j, θ_i) = exp(θ_{i,1} * j + θ_{i,2} * j²) / Σ_{j=0}^{K_i} exp(θ_{i,1} * j + θ_{i,2} * j²).
    """

    def __init__(self, y: pd.Series, x_list: list, K_list: list):
        """
        Paramètres
        ----------
        y : pd.Series
            Variable dépendante.
        x_list : list de pd.Series
            Liste des variables macro explicatives.
        K_list : list d'entiers
            Liste du nombre de retards pour chaque variable macro.
        """
        self.y = y.copy()
        self.x_list = [x.copy() for x in x_list]
        self.K_list = K_list
        self.params = None

        # Aligner toutes les séries sur leur index commun
        common_index = self.y.index
        for x in self.x_list:
            common_index = common_index.intersection(x.index)
        self.y = self.y.loc[common_index]
        self.x_list = [x.loc[common_index] for x in self.x_list]

    def almon_weights(self, theta1, theta2, K):
        """
        Calcule les poids pour un nombre de retards K avec une fonction exponentielle Almon.
        """
        j = np.arange(K + 1)
        w = np.exp(theta1 * j + theta2 * j ** 2)
        return w / np.sum(w)

    def midas_prediction(self, params):
        """
        Calcule les prédictions du modèle pour l'ensemble des observations.
        Les paramètres sont organisés de la façon suivante :
            [alpha, beta1, theta11, theta12, beta2, theta21, theta22, ..., beta_M, theta_M1, theta_M2]
        """
        alpha = params[0]
        n = len(self.y)
        pred = np.zeros(n)
        offset = 1

        # Pour chaque variable macro, on calcule la contribution
        for i, K in enumerate(self.K_list):
            beta = params[offset]
            theta1 = params[offset + 1]
            theta2 = params[offset + 2]
            offset += 3

            w = self.almon_weights(theta1, theta2, K)
            xi = self.x_list[i]
            preds_i = np.full(n, np.nan)
            # On calcule les prédictions à partir de l'observation K
            for t in range(K, n):
                x_lags = xi.iloc[t - K:t + 1].values[::-1] # x_{i,t}, x_{i,t-1}, ..., x_{i,t-K}
                preds_i[t] = np.dot(w, x_lags)
            # Ajout de la contribution de la variable i
            pred = np.where(~np.isnan(preds_i), pred + beta * preds_i, pred)

        # Les premières observations où tous les retards ne sont pas disponibles sont ignorées
        max_K = max(self.K_list)
        return alpha + pred[max_K:], max_K

    def objective(self, params):
        """
        Fonction objectif : différences entre les y observés et les y prédits.
        """
        y_obs = self.y.iloc[max(self.K_list):].values
        y_pred, _ = self.midas_prediction(params)
        return y_pred - y_obs

    def fit(self, initial_guess=None):
        """
        Estime les paramètres du modèle.
        L'estimation initiale par défaut est [alpha, puis pour chaque variable : beta, theta1, theta2].
        """
        if initial_guess is None:
            # Initialisation : alpha=0, puis pour chaque variable : beta=0.1, theta1=0, theta2=0
            initial_guess = [0] + [0.1, 0, 0] * len(self.x_list)
        res = least_squares(self.objective, initial_guess)
        self.params = res.x
        return self.params

    def predict(self):
        """
        Retourne les prédictions du modèle à partir des paramètres estimés.
        """
        if self.params is None:
            raise Exception("Le modèle n'a pas encore été ajusté. Utilisez la méthode fit() d'abord.")
        y_pred, offset = self.midas_prediction(self.params)
        return y_pred

    def plot_fit(self):
        """
        Trace les observations et les prédictions du modèle.
        """
        if self.params is None:
            raise Exception("Le modèle n'a pas encore été ajusté. Utilisez la méthode fit() d'abord.")
        y_obs = self.y.iloc[max(self.K_list):]
        y_pred = self.predict()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_obs.index, y=y_obs.values,
                                 mode='lines', name='Observé'))
        fig.add_trace(go.Scatter(x=y_obs.index, y=y_pred,
                                 mode='lines', name='Prédit'))
        fig.update_layout(title="Ajustement du modèle Multi-MIDAS",
                          xaxis_title="Date",
                          yaxis_title="y",
                          hovermode="x")
        fig.show()

