# cashflow/project.py
import pandas as pd
from config import CATEGORIES
from .base import BaseCashflowEngine


class ProjectCashflow(BaseCashflowEngine):
    """Handles project-specific cash flow calculations."""

    def __init__(
            self,
            bank_statement_df,
            acquisition_date,
            start_date,
            end_date,
            forecast_periods_count,
            development_cost_adjustment,
            input_assumptions,
            additional_unit_cost,
            proposed_budget_data=None,
            unexpected_costs=0.0
    ):
        super().__init__(acquisition_date, start_date, end_date)

        self.bank_statement_df = bank_statement_df.copy()
        self.development_cost_adjustment = development_cost_adjustment
        self.forecast_periods_count = forecast_periods_count
        self.additional_unit_cost = additional_unit_cost
        self.unexpected_costs = unexpected_costs

        # Generate forecast periods
        self.forecast_periods = pd.date_range(
            start=self.date_processor.start_date,
            periods=forecast_periods_count,
            freq='ME'
        )

        # Initialize assumptions
        self.input_assumptions = input_assumptions
        self.proposed_budget_data = proposed_budget_data

        # Initialize the cashflow DataFrame
        self.cashflow_df = self.initialize_dataframe(CATEGORIES['all'])

    def populate_historical_costs(self):
        """Populate historical costs from bank statement data."""
        for period in self.date_processor.all_periods:
            if period > self.date_processor.start_date:
                continue

            period_str = period.strftime('%b-%y')
            for category in CATEGORIES['historical']:
                historical_data = self.bank_statement_df[
                    (self.bank_statement_df['TAG'] == category) &
                    (self.bank_statement_df['MONTH'] == period)
                    ]
                amount = historical_data['MOVEMENT'].sum()

                self.cashflow_df.loc[category, period_str] = amount

    def calculate_remaining_budget(self):
        """Calculate remaining budget after accounting for Antvic transactions."""
        budget_diff = (
                self.proposed_budget_data['revised_budget'] -
                self.proposed_budget_data['pre_antvic_budget']
        )

        antvic_transactions = self.bank_statement_df[
            self.bank_statement_df['DETAIL 1'].str.lower().str.contains('antvic', na=False)
        ]['MOVEMENT'].sum()

        return budget_diff - abs(antvic_transactions)

    def calculate_monthly_development_cost(self):
        """Calculate monthly development cost for forecasting."""
        if self.forecast_periods_count <= 0:
            return 0.0

        remaining_budget = self.calculate_remaining_budget()

        total_costs = (
                remaining_budget +
                self.additional_unit_cost +
                self.input_assumptions['Direct Fees'] +
                self.input_assumptions['Other Project Costs'] +
                self.unexpected_costs
        )

        return total_costs / self.forecast_periods_count

    def forecast_development_costs(self):
        """Forecast development costs across forecast periods."""
        monthly_cost = self.calculate_monthly_development_cost()

        for period in self.forecast_periods:
            period_str = period.strftime('%b-%y')
            self.cashflow_df.loc['Development costs', period_str] = -monthly_cost

    def forecast_other_costs(self):
        """Forecast other development costs."""
        QUARTERLY_MONTH_INDEX = 2  # Third month in quarter (0-based index)

        for idx, period in enumerate(self.forecast_periods):
            period_str = period.strftime('%b-%y')

            for category in CATEGORIES['development']:
                if category in {'Development costs', 'Management Fee'}:
                    continue

                cost = self.input_assumptions.get(category, 0.0)
                if cost <= 0:
                    continue

                if category in CATEGORIES['quarterly']:
                    if idx % 3 == QUARTERLY_MONTH_INDEX:
                        self.cashflow_df.loc[category, period_str] = -cost
                else:
                    self.cashflow_df.loc[category, period_str] = -cost

    def forecast_management_fee(self):
        """Forecast Management Fee with special calculation for last period."""
        # Calculate Max TDC
        dev_categories = [
            'Accountancy', 'Planning & Design', 'Development costs',
            'Legal & Professional', 'Insurance', 'VAT',
            'Additional', 'Operation'
        ]

        max_tdc = self.cashflow_df.loc[dev_categories, 'Total'].sum()
        max_tdc += self.cashflow_df.loc['Acquisition Total', 'Total']

        # Forecast standard fee for periods except the last
        fee = self.input_assumptions.get('Management Fee', 0.0)

        for idx, period in enumerate(self.forecast_periods):
            if idx == self.forecast_periods_count - 1:  # Skip last period
                continue

            if idx == 2:  # Special case for third period
                period_str = period.strftime('%b-%y')
                self.cashflow_df.loc['Management Fee', period_str] = -fee

        # Calculate last period fee
        acq_date_str = self.date_processor.acquisition_date.strftime('%b-%y')
        total_fee_so_far = self.cashflow_df.loc['Management Fee', acq_date_str:][:-1].sum()

        last_period_str = self.forecast_periods[-1].strftime('%b-%y')
        last_period_fee = (max_tdc * 0.01) - total_fee_so_far
        self.cashflow_df.loc['Management Fee', last_period_str] = last_period_fee

    def calculate_totals(self):
        """Calculate category totals for each period."""
        for period_str in self.date_processor.monthly_period_strs:
            # Development total
            self.cashflow_df.loc['Development', period_str] = self.cashflow_df.loc[
                CATEGORIES['development_components'], period_str
            ].sum()

            # Acquisition total
            self.cashflow_df.loc['Acquisition Total', period_str] = self.cashflow_df.loc[
                CATEGORIES['acquisition_components'], period_str
            ].sum()

            # Project total
            self.cashflow_df.loc['Total Project Cashflow', period_str] = self.cashflow_df.loc[
                CATEGORIES['total_project_components'], period_str
            ].sum()

            # Total Development Costs
            self.cashflow_df.loc['Total Development Costs', period_str] = (
                    self.cashflow_df.loc['Development', period_str] +
                    self.cashflow_df.loc['Financing costs', period_str] +
                    self.cashflow_df.loc['Acquisition Total', period_str]
            )

    def generate_cashflow(self):
        """Generate the complete project cashflow."""
        # Process historical data
        self.populate_historical_costs()

        # Process forecasts
        self.forecast_development_costs()
        self.forecast_other_costs()

        # Calculate interim totals
        self.calculate_totals()

        # Calculate management fee (needs totals)
        self.forecast_management_fee()

        # Recalculate totals with management fee
        self.calculate_totals()

        # Calculate period summaries
        self.calculate_all_period_totals()

        return self.cashflow_df.round(2)