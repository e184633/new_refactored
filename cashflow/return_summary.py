# Create a new file: cashflow/summary.py
import pandas as pd


class ReturnSummaryCalculator:
    """Centralizes calculations for financial summaries and returns."""

    def __init__(self, cashflow_df, config):
        self.cashflow_df = cashflow_df
        self.config = config

        # Calculate and store key values once
        self._calculate_values()

    def _calculate_values(self):
        """Calculate and store all key financial metrics."""
        # Exit value calculations
        self.exit_amount = self.config.get('input_assumptions', {}).get('exit_amount', 79500000)
        self.selling_costs_pct = self.config.get('selling_costs_pct', 0.002)
        self.selling_costs = -self.exit_amount * self.selling_costs_pct
        self.net_sale_price = self.exit_amount + self.selling_costs

        # Cost calculations
        self.acquisition_total = 0
        self.development_costs = 0
        self.financing_costs = 0
        self.total_development_cost = 0

        if self.cashflow_df is not None and not self.cashflow_df.empty:
            for idx, row in self.cashflow_df.iterrows():
                category = row.get('Category', row.name)
                if category == 'Acquisition Total' and 'Total' in self.cashflow_df.columns:
                    self.acquisition_total = row['Total']
                elif category == 'Development' and 'Total' in self.cashflow_df.columns:
                    self.development_costs = row['Total']
                elif category == 'Financing costs' and 'Total' in self.cashflow_df.columns:
                    self.financing_costs = row['Total']
                elif category == 'Total Development Costs' and 'Total' in self.cashflow_df.columns:
                    self.total_development_cost = row['Total']

    def get_exit_value_df(self):
        """Return a DataFrame for the Exit Value table."""
        exit_data = {
            "Item": ["Sell price", "Selling Costs", "Net sale price"],
            "Value": [
                f"£{self.exit_amount:,.0f}",
                f"(£{abs(self.selling_costs):,.0f})",
                f"£{self.net_sale_price:,.0f}"
            ]
        }
        return pd.DataFrame(exit_data).set_index("Item")

    def get_costs_df(self):
        """Return a DataFrame for the Costs table."""
        costs_data = {
            "Item": ["Acquisition", "Development", "Financing costs", "Total Development cost"],
            "Value": [
                f"(£{abs(self.acquisition_total):,.0f})" if self.acquisition_total < 0 else f"£{self.acquisition_total:,.0f}",
                f"(£{abs(self.development_costs):,.0f})" if self.development_costs < 0 else f"£{self.development_costs:,.0f}",
                f"(£{abs(self.financing_costs):,.0f})" if self.financing_costs < 0 else f"£{self.financing_costs:,.0f}",
                f"(£{abs(self.total_development_cost):,.0f})" if self.total_development_cost < 0 else f"£{self.total_development_cost:,.0f}"
            ]
        }
        return pd.DataFrame(costs_data).set_index("Item")

    def get_profit(self):
        """Calculate profit (net sale price minus total development cost)."""
        return self.net_sale_price + self.total_development_cost  # Adding a negative number

    # Add more methods for the profit split calculations as needed
