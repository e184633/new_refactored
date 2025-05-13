# cashflow/manager.py
from .debt import DebtCashflow
from .project import ProjectCashflow


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
            unexpected_costs=0.0
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
            unexpected_costs=unexpected_costs
        )

        # Create debt cashflow processor
        self.debt_cashflow = DebtCashflow(
            senior_loan_statement_df=senior_loan_statement_df,
            mezzanine_loan_statement_df=mezzanine_loan_statement_df,
            acquisition_date=acquisition_date,
            start_date=start_date,
            end_date=end_date,
            annual_base_rate=annual_base_rate
        )

        # Store primary parameters
        self.acquisition_date = acquisition_date
        self.start_date = start_date
        self.end_date = end_date
        self.annual_base_rate = annual_base_rate

    def generate_cashflow(self):
        """Generate project cashflow."""
        return self.project_cashflow.generate_cashflow()

    def generate_debt_cashflow(self):
        """Generate debt cashflow and store it as an attribute."""
        self.debt_cashflow_df = self.debt_cashflow.generate_cashflow()
        return self.debt_cashflow_df
