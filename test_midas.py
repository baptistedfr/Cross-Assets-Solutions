from Midas.midas import MIDASModel, MultiMIDASModel
import pandas as pd
import plotly.graph_objects as go

# Exemple d'utilisation
# if __name__ == "__main__":
#     # Chargement des données pour le secteur et pour les variables macro
#     data = pd.read_excel("datas/Stoxx600_sectors_prices_cleen.xlsx", index_col=0)
#     ten_year_yield = pd.read_csv("data_macro/ten_year_bond_yield.csv", index_col=0, parse_dates=True)
#     refi_rate = pd.read_csv("data_macro/refinancing_rate.csv", index_col=0, parse_dates=True)
#     cpi = pd.read_csv("data_macro/cpi.csv", index_col=0, parse_dates=True)
#     cpi = cpi["TACECO/CPIYOY/EUZ"]
#     ten_year_yield = ten_year_yield.resample("M").last()
#     cpi = cpi.resample("M").last()
#
#     for sect in data.columns:
#         secteur = data[sect]
#         rendement_secteur = secteur.pct_change().dropna()
#         rendement_secteur.name = "Secteur " + data.columns[0]
#         rendement_secteur = rendement_secteur.resample("M").last()
#         # Variables macro
#         taux10y = ten_year_yield["rate"]
#
#         taux10y.name = "10Y German Bond Rate"
#         cpi.name = "CPI Monthly Rate"
#
#         # Alignement sur les dates communes
#         rendement_secteur, taux10y = rendement_secteur.align(ten_year_yield, join="inner", axis=0)
#         rendement_secteur, cpi = rendement_secteur.align(cpi, join="inner", axis=0)
#
#         # Création du modèle Multi-MIDAS avec 2 variables macro et des nombres de retards spécifiques
#         multi_midas = MultiMIDASModel(rendement_secteur, [taux10y, cpi], K_list=[5, 3])
#         params_estimes = multi_midas.fit()
#         print("Paramètres estimés :", params_estimes)
#
#         # Visualisation de l'ajustement du modèle
#         multi_midas.plot_fit()

# Exemple d'utilisation
if __name__ == "__main__":
    # Chargement des données des secteurs et des variables macro
    data = pd.read_excel("datas/Stoxx600_sectors_prices_cleen.xlsx", index_col=0)
    market = pd.read_excel("datas/stoxx600.xlsx", index_col=0)
    ten_year_yield = pd.read_csv("data_macro/ten_year_bond_yield.csv", index_col=0, parse_dates=True)
    refi_rate = pd.read_csv("data_macro/refinancing_rate.csv", index_col=0, parse_dates=True)
    cpi = pd.read_csv("data_macro/cpi.csv", index_col=0, parse_dates=True)
    zew = pd.read_excel("data_macro/A REQUEST.xlsx", index_col=0, sheet_name="GRZECURR Index")
    unemployment = pd.read_excel("data_macro/A REQUEST.xlsx", index_col=0, sheet_name="UMRTEMU  Index")
    savings_rate = pd.read_excel("data_macro/A REQUEST.xlsx", index_col=0, sheet_name="EUESEMU Index")

    # Préparation des variables macro
    taux10y = ten_year_yield["rate"]
    refi_series = refi_rate["ECBDFR"].dropna()
    refi_series = refi_series.resample("M").last()
    cpi_rate = cpi["TACECO/CPIYOY/EUZ"].dropna()
    zew_series = zew.astype(float)["value"]
    unemployment_series = unemployment.astype(float)["value"]
    savings_series = savings_rate.astype(float)["value"]

    # Création d'un dictionnaire regroupant les séries macro
    macro_vars = {
        "Taux10Y": taux10y,
        "Refi Rate": refi_series,
        "CPI Rate": cpi_rate,
        "Zew": zew_series,
        "Unemployment": unemployment_series,
        "Savings Rate": savings_series
    }

    # Pour chaque variable macro et pour chaque secteur, ajuster un modèle MIDAS
    # Dictionary to store beta values for each macro variable.
    all_betas = {}

    for macro_name, macro_series in macro_vars.items():
        print(f"\n--- Testing the MIDAS model for macro variable: {macro_name} ---")

        # Create an empty dictionary to collect beta values for each sector for this macro variable.
        beta_dict = {}

        for sect in data.columns:
            secteur = data[sect]
            rendement_secteur = secteur.pct_change().dropna()

            # Align the dates between the sector return and the macro variable
            aligned_sector, aligned_macro = rendement_secteur.align(macro_series, join="inner", axis=0)
            if aligned_sector.empty or aligned_macro.empty:
                print(f"No aligned data for sector {sect} with macro variable {macro_name}.")
                continue

            aligned_sector.name = sect
            aligned_macro.name = macro_name


            # Create and fit the MIDAS model (assuming MIDASModel is already defined)
            midas_model = MIDASModel(aligned_sector, aligned_macro, K=12)
            params_estimes = midas_model.fit()
            print(f"Sector: {sect} | Macro: {macro_name} | Estimated parameters: {params_estimes}")

            # plot the fit
            # midas_model.plot_fit()

            # Extract the beta estimate from the MIDAS model parameters.
            beta_sect = params_estimes['beta']
            beta_dict[sect] = beta_sect

        beta_df = pd.DataFrame.from_dict(beta_dict, orient="index", columns=["Beta"])
        all_betas[macro_name] = beta_df
        all_betas[macro_name].to_csv(f"results/betas_{macro_name}.csv")


