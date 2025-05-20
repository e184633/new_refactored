# cashflow/manager.py
from .debt import DebtCashflow
from .project import ProjectCashflow
from config import DEFAULT_CONFIG
from .equity import EquityCashflow


class CashflowManager:
    """Coordinates different cashflow processors and provides a unified interface."""

    def __init__(
            self,
            # Core data inputs
            bank_statement_df,
            senior_loan_statement_df,
            mezzanine_loan_statement_df,

            # Date parameters
            acquisition_date,
            start_date,
            end_date,

            # Forecast configuration
            forecast_periods_count,
            development_cost_adjustment,
            annual_base_rate,

            # Financial assumptions
            input_assumptions,
            additional_unit_cost,
            proposed_budget_data=None,
            unexpected_costs=0.0,
            additional_advertisement_cost=DEFAULT_CONFIG['additional_advertisement_cost'],
    ):
        # Create project cashflow processor
        self.project_cashflow = ProjectCashflow(
            bank_statement_df=bank_statement_df,
            acquisition_date=acquisition_date,
            start_date=start_date,
            end_date=end_date,
            forecast_periods_count=forecast_periods_count,
            development_cost_adjustment=development_cost_adjustment,
            input_assumptions=input_assumptions,
            additional_unit_cost=additional_unit_cost,
            proposed_budget_data=proposed_budget_data,
            unexpected_costs=unexpected_costs,
            additional_advertisement_cost=additional_advertisement_cost
        )

        # Create debt cashflow processor
        self.debt_cashflow = DebtCashflow(
            senior_loan_statement_df=senior_loan_statement_df,
            mezzanine_loan_statement_df=mezzanine_loan_statement_df,
            acquisition_date=acquisition_date,
            start_date=start_date,
            end_date=end_date,
            annual_base_rate=annual_base_rate,
            forecast_periods_count=forecast_periods_count,
        )
        # Create equity cashflow processor
        self.equity_cashflow = EquityCashflow(
            bank_statement_df=bank_statement_df,
            acquisition_date=acquisition_date,
            start_date=start_date,
            end_date=end_date,
            forecast_periods_count=forecast_periods_count,
            input_assumptions=input_assumptions,

        )

        # Store primary parameters
        self.acquisition_date = acquisition_date
        self.start_date = start_date
        self.end_date = end_date
        self.annual_base_rate = annual_base_rate

    def calculate_equity_split_percentages(self):
        """Calculate equity split percentages for visualization."""
        equity_split = {}
        shareholders = self.equity_cashflow.get_shareholder_names()

        # Look for shareholder rows and total equity in equity_cashflow_df
        total_equity = 0
        for idx, row in self.equity_cashflow_df.iterrows():
            if row['Category'] == 'Total Shareholder Capital' and 'Total' in self.equity_cashflow_df.columns:
                total_value = row['Total']
                if isinstance(total_value, (float, int)):
                    total_equity = total_value

        # Get each shareholder's total
        for shareholder in shareholders:
            for idx, row in self.equity_cashflow_df.iterrows():
                if row['Category'] == shareholder and 'Total' in self.equity_cashflow_df.columns:
                    value = row['Total']
                    if isinstance(value, (float, int)) and total_equity > 0:
                        equity_split[shareholder] = (value / total_equity) * 100

        return equity_split

    def calculate_combined_financing_costs(self):
        """Calculate combined financing costs from both loans for each period."""
        financing_costs = {}

        # Categories to sum for financing costs
        financing_categories = ["Fees", "Capitalised Interest", "Non-utilisation Fee", "IMS Fees"]

        # Find all rows for these categories in debt cashflow
        for idx, row in self.debt_cashflow_df.iterrows():
            category = row.get("Category")
            if category not in financing_categories:
                continue
            # Sum for each period
            for col in self.debt_cashflow_df.columns:
                if col == "Category" and col in self.debt_cashflow.date_processor.overview_columns:
                    continue
                period = col
                if period not in financing_costs:
                    financing_costs[period] = 0.0

                if isinstance(row[col], (int, float)):
                    financing_costs[period] += row[col]

        return financing_costs

    def generate_cashflow(self):
        """Generate project cashflow with financing costs from debt data."""
        # First generate the debt cashflow to calculate financing costs
        self.generate_debt_cashflow()

        # Now extract financing costs from debt data
        financing_costs = self.calculate_combined_financing_costs()

        # Pass financing costs to project cashflow
        self.project_cashflow.set_financing_costs(financing_costs)

        # Generate the project cashflow
        return self.project_cashflow.generate_cashflow()

    def generate_debt_cashflow(self):
        """Generate debt cashflow and store it as an attribute."""
        self.debt_cashflow_df = self.debt_cashflow.generate_cashflow()
        return self.debt_cashflow_df

    def generate_equity_cashflow(self):
        """Generate equity cashflow."""
        # Generate project cashflow if not already generated
        if not hasattr(self, 'project_cashflow_df'):
            self.project_cashflow_df = self.project_cashflow.generate_cashflow()

        # Generate debt cashflow if not already generated
        if not hasattr(self, 'debt_cashflow_df'):
            self.debt_cashflow_df = self.debt_cashflow.generate_cashflow()

        # Create equity cashflow with access to project and debt data
        self.equity_cashflow.project_cashflow_df = self.project_cashflow_df
        self.equity_cashflow.debt_cashflow_df = self.debt_cashflow_df

        self.equity_cashflow_df = self.equity_cashflow.generate_cashflow()
        self.equity_split_data = self.calculate_equity_split_percentages()
        return self.equity_cashflow_df
