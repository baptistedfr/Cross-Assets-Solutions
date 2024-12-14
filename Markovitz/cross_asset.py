import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def calculate_var(weights, mean_returns, cov_matrix, confidence_level=0.95):
    portfolio_mean = np.sum(mean_returns * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    z_score = np.abs(np.percentile(np.random.normal(0, 1, 100000), (1 - confidence_level) * 100))
    var = portfolio_mean - z_score * portfolio_std
    return -var  # Negative since we minimize risk

def calculate_cvar(weights, mean_returns, cov_matrix, confidence_level=0.95):
    portfolio_mean = np.sum(mean_returns * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    z_score = np.abs(np.percentile(np.random.normal(0, 1, 100000), (1 - confidence_level) * 100))
    simulated_losses = np.random.normal(portfolio_mean, portfolio_std, 100000)
    var_threshold = np.percentile(simulated_losses, (1 - confidence_level) * 100)
    cvar = -np.mean(simulated_losses[simulated_losses <= var_threshold])
    return cvar

def random_portfolios(num_portfolios, mean_returns, cov_matrix, confidence_level=0.95):
    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)

        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        var = calculate_var(weights, mean_returns, cov_matrix, confidence_level)
        # cvar = calculate_cvar(weights, mean_returns, cov_matrix, confidence_level)

        results[0, i] = var  # Value at Risk
        results[1, i] = portfolio_return
        results[2, i] = portfolio_return / (var if var != 0 else np.nan)  # Return-to-VaR ratio
        # results[3, i] = cvar

    return results, weights_record


def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, confidence_level=0.95):
    results, weights = random_portfolios(num_portfolios, mean_returns, cov_matrix, confidence_level)

    # Portefeuille avec le maximum Return-to-VaR ratio
    max_risk_adj_idx = np.nanargmax(results[2])  # Max Return-to-VaR ratio
    best_var, best_return = results[0, max_risk_adj_idx], results[1, max_risk_adj_idx]
    best_weights = weights[max_risk_adj_idx]

    # Affichage de l'allocation du portefeuille optimal pour Return-to-VaR
    best_allocation = pd.DataFrame(best_weights, index=mean_returns.index, columns=['allocation'])
    best_allocation.allocation = [round(i * 100, 2) for i in best_allocation.allocation]

    print("Portfolio Allocation Optimized for Maximum Return-to-VaR Ratio\n")
    print("Annualised Return:", round(best_return, 2))
    print("Value at Risk (95%):", round(best_var, 2))
    print(best_allocation)

    # Portefeuille avec la VaR minimale
    min_var_idx = np.nanargmin(results[0])  # Min VaR
    min_var, min_return = results[0, min_var_idx], results[1, min_var_idx]
    min_var_weights = weights[min_var_idx]

    # Affichage de l'allocation du portefeuille optimal pour la VaR minimale
    min_var_allocation = pd.DataFrame(min_var_weights, index=mean_returns.index, columns=['allocation'])
    min_var_allocation.allocation = [round(i * 100, 2) for i in min_var_allocation.allocation]

    print("\nPortfolio Allocation Optimized for Minimum VaR\n")
    print("Annualised Return:", round(min_return, 2))
    print("Value at Risk (95%):", round(min_var, 2))
    print(min_var_allocation)

    # min_cvar_idx = np.nanargmin(results[3])  # Min CVaR
    # min_cvar, min_cvar_return = results[3, min_cvar_idx], results[1, min_cvar_idx]
    # min_cvar_weights = weights[min_cvar_idx]
    #
    # min_cvar_allocation = pd.DataFrame(min_cvar_weights, index=mean_returns.index, columns=['allocation'])
    # min_cvar_allocation.allocation = [round(i * 100, 2) for i in min_cvar_allocation.allocation]

    # Création du graphique
    plt.figure(figsize=(12, 8))
    # Nuage de points pour les portefeuilles simulés
    plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='YlGnBu', marker='o', s=10, alpha=0.4)
    plt.colorbar(label='Return-to-VaR Ratio')

    # Portefeuille avec le max Return-to-VaR ratio
    plt.scatter(best_var, best_return, marker='*', color='r', s=500, label='Max Return-to-VaR')

    # Portefeuille avec la VaR minimale
    plt.scatter(min_var, min_return, marker='*', color='b', s=500, label='Min VaR')

    # plt.scatter(results[3, :], results[1, :], color='purple', alpha=0.3, label='CVaR Points')
    # plt.scatter(min_cvar, min_cvar_return, marker='*', color='g', s=500, label='Optimal Portfolio (Min CVaR)')
    # Titre et légendes
    plt.title('Simulated Portfolios Based on Value at Risk (VaR 95%)', fontsize=16)
    plt.xlabel('Value at Risk (95%)', fontsize=12)
    plt.ylabel('Annualised Returns', fontsize=12)
    plt.legend(labelspacing=0.8)
    plt.grid(True)
    plt.show()

    return min_var_allocation

# Lecture des données
df = pd.read_csv('prices.csv', sep=';', index_col=0)
df = df.iloc[1:]
df = df.apply(lambda x: x.str.replace(',', '.'))
df = df.apply(pd.to_numeric, errors='coerce')
df.ffill(inplace=True)
df.dropna(inplace=True)

returns = df.pct_change()
returns = returns.dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 10000
confidence_level = 0.95

display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, confidence_level)

print('done')