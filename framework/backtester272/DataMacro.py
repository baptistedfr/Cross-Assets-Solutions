import pandas as pd
import os

class DataMacro:

    def __init__(self):
        self.base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data_macro"))
        self.load_macro_data()
        self.load_sensitivity_matrix()

    def load_macro_data(self, 
                        spread_2Y10Y_file: str = "spread_2Y10Y.csv",
                        yield_10Y_file: str = "ten_year_bond_yield.csv",
                        cpi_file: str = "cpi.csv", 
                        #refi_rate_file: str = "refinancing_rate.csv", 
                        ZEW_file: str = "zew.csv",
                        unemployment_file: str = "unemployment_rate.csv",
                        savings_file: str = "savings_rate.csv",
                        GDP_file: str = "gdp.csv"):
        """
        Load and format macroeconomic data (CPI, refinancing rate, and ten-year bond yield).
        - Ensures the date is set as the index and sorted.
        """
        # Load macroeconomic data
        spread_2Y10Y_path = os.path.join(self.base_path, spread_2Y10Y_file)
        yield_10Y_path = os.path.join(self.base_path, yield_10Y_file)
        cpi_path = os.path.join(self.base_path, cpi_file)
        ZEW_path = os.path.join(self.base_path, ZEW_file)
        unemployment_path = os.path.join(self.base_path, unemployment_file)
        savings_path = os.path.join(self.base_path, savings_file)
        GDP_path = os.path.join(self.base_path, GDP_file)

        # Load data into pandas DataFrames
        self.spread_2Y10Y = pd.read_csv(spread_2Y10Y_path, index_col=0, parse_dates=True).sort_index().astype(float).pct_change()[1:]
        self.yield_10Y = pd.read_csv(yield_10Y_path, index_col=0, parse_dates=True).sort_index().astype(float).pct_change()[1:]
        self.cpi = pd.read_csv(cpi_path, index_col=0, parse_dates=True).sort_index().astype(float).pct_change()[1:]
        self.ZEW = pd.read_csv(ZEW_path, index_col=0, parse_dates=True).sort_index().astype(float).pct_change()[1:]
        self.unemployment = pd.read_csv(unemployment_path, index_col=0, parse_dates=True).sort_index().astype(float).pct_change()[1:]
        self.savings = pd.read_csv(savings_path, index_col=0, parse_dates=True).sort_index().astype(float).pct_change()[1:]
        self.GDP = pd.read_csv(GDP_path, index_col=0, parse_dates=True).sort_index().astype(float).pct_change()[1:]

        # cpi_path = os.path.join(self.base_path, cpi_file)
        # refi_rate_path = os.path.join(self.base_path, refi_rate_file)
        # yield_10Y_path = os.path.join(self.base_path, yield_10Y_file)

        # self.cpi = pd.read_csv(cpi_path, index_col=0, parse_dates=True).sort_index().astype(float).pct_change()[1:]
        # self.refi_rate = pd.read_csv(refi_rate_path, index_col=0, parse_dates=True).sort_index().astype(float).pct_change()[1:]
        # self.yield_10Y = pd.read_csv(yield_10Y_path, index_col=0, parse_dates=True).sort_index().astype(float).pct_change()[1:]

    def load_sensitivity_matrix(self, increases_file: str = "sensi_decreases.csv", decreases_file: str = "sensi_increases.csv"):
        """
        Load the sensitivity matrix for sectors and macro variables.
        - The matrix describes the sensitivity of each sector to macroeconomic variables.
        """
        increases_path = os.path.join(self.base_path, increases_file)
        decreases_path = os.path.join(self.base_path, decreases_file)
        self.increases = pd.read_csv(increases_path, index_col=0).astype(float)
        self.decreases = pd.read_csv(decreases_path, index_col=0).astype(float)

    def compute_weighted_view(self, macro_view: pd.Series) -> pd.Series:
        """
        Compute the weighted macro view based on sensitivities.
        - Normalizes the macro views to avoid scale bias across variables.
        - Multiplies the normalized macro views by the sensitivity matrix to compute sector-level impacts.

        :param macro_view: series with next estimated value for each macro variable (% > average for example)
                           Keys should match the columns of the sensitivity matrix.
        :return: pandas Series with tilts for each sector.
        """
        if not set(macro_view.index).issubset(self.increases.columns):
            raise ValueError("Macro view keys must match sensitivity matrix columns.")
        
        if not set(macro_view.index).issubset(self.decreases.columns):
            raise ValueError("Macro view keys must match sensitivity matrix columns.")
        
        # Normalize macro views to avoid scale bias
        normalized_macro_view = (macro_view - macro_view.mean()) / macro_view.std()
        
        # Compute tilts by applying different sensitivity matrices for up and down values
        tilts = pd.Series(0, index=self.increases.index)
        for macro in normalized_macro_view.index:
            if normalized_macro_view[macro] >= 0:
                tilts += self.increases[macro] * normalized_macro_view[macro]
            else:
                tilts += self.decreases[macro] * normalized_macro_view[macro]
        
        return tilts