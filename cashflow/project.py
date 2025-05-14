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
            unexpected_costs=0.0,
            additional_advertisement_cost=None,
    ):
        super().__init__(acquisition_date, start_date, end_date, forecast_periods_count)

        self.bank_statement_df = bank_statement_df.copy()
        self.development_cost_adjustment = development_cost_adjustment
        self.forecast_periods_count = forecast_periods_count
        self.additional_unit_cost = additional_unit_cost
        self.unexpected_costs = unexpected_costs
        self.additional_advertisement_cost = additional_advertisement_cost

        # Generate forecast periods
        self.forecast_periods = pd.date_range(
            start=self.date_processor.current_date,
            periods=forecast_periods_count,
            freq='ME'
        )

        # Initialize assumptions
        self.input_assumptions = input_assumptions
        self.proposed_budget_data = proposed_budget_data

        # Initialize the cashflow DataFrame
        self.cashflow_df = self.initialise_dataframe(CATEGORIES['all'])

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
        forecast_periods = self.date_processor.filter_periods_until_date(self.forecast_periods)
        return total_costs / len(forecast_periods)

    def forecast_development_costs(self):
        """Forecast development costs across forecast periods."""
        monthly_cost = self.calculate_monthly_development_cost()

        forecast_periods = self.date_processor.filter_periods_until_date(self.forecast_periods)

        for period in forecast_periods:
            period_str = period.strftime('%b-%y')
            self.cashflow_df.loc['Development costs', period_str] = -monthly_cost

    def forecast_other_costs(self):
        """Forecast other development costs."""
        QUARTERLY_MONTH_INDEX = 2  # Third month in quarter (0-based index)
        forecast_periods = self.date_processor.filter_periods_until_date(self.forecast_periods)

        for idx, period in enumerate(forecast_periods):
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
            'Additional Advertisement', 'Operation'
        ]
        forecast_periods = self.date_processor.filter_periods_until_date(self.forecast_periods)

        max_tdc = self.cashflow_df.loc[dev_categories, 'Total'].sum()
        # print('MAX TDC', max_tdc)
        max_tdc += self.cashflow_df.loc['Acquisition Total', 'Total']
        max_tdc += self.cashflow_df.loc['Financing costs', 'Total']  * .5

        # Forecast standard fee for periods except the last
        fee = self.input_assumptions.get('Management Fee', 0.0)

        for idx, period in enumerate(forecast_periods):
            if idx == self.forecast_periods_count - 1:  # Skip last period
                continue

            if idx == 2:  # Special case for third period
                period_str = period.strftime('%b-%y')
                self.cashflow_df.loc['Management Fee', period_str] = -fee

        # Calculate last period fee
        total_fee_so_far = self.cashflow_df.loc['Management Fee', :].sum()

        last_period_str = self.forecast_periods[-1].strftime('%b-%y')
        # print(last_period_str, self.cashflow_df.loc['Management Fee', :].sum(), max_tdc * 0.01)
        last_period_fee = (max_tdc * 0.01) - total_fee_so_far
        self.cashflow_df.loc['Management Fee', last_period_str] = last_period_fee

    def populate_financing_costs(self):
        """Populate financing costs from debt analysis."""
        if hasattr(self, 'financing_costs') and self.financing_costs:
            # Check if 'financing costs' or 'Financing costs' exists in the index
            category = 'Financing costs'

            # Now populate the values
            for period, amount in self.financing_costs.items():
                if period in self.cashflow_df.columns:
                    self.cashflow_df.loc[category, period] = amount

    def set_financing_costs(self, financing_costs):
        """Set financing costs from debt analysis."""
        self.financing_costs = financing_costs

    def forecast_additional_advertisement_cost(self):
        """Forecast additional unit cost across forecast periods."""
        # Determine target category
        target_category = None

        # Look for the "Additional Advertisement" row
        target_category = "Additional Advertisement"
        periods = self.date_processor.filter_periods('start_to_exit')
        print("PERIODS ", len(periods), self.date_processor.start_date)
        # Calculate monthly cost
        monthly_cost = self.additional_advertisement_cost / len(periods) if self.forecast_periods_count > 0 else 0

        # Distribute across forecast periods
        for period in periods:
            period_str = period.strftime('%b-%y')
            self.cashflow_df.loc[target_category, period_str] = -monthly_cost

    # In cashflow/project.py, update the forecast_additional_unit_cost method:

    def forecast_additional_unit_cost(self):
        """Forecast additional unit cost across forecast periods."""
        # Find the target category
        target_category = "Additional"
        for category in self.cashflow_df.index:
            if "additional advertisement" in category.lower():
                target_category = category
                break

        # Only forecast until exit date
        # print(self.forecast_periods)
        forecast_periods = self.date_processor.filter_periods(period_type='start_to_exit')
        print(forecast_periods, self.date_processor.start_date, 'START_DATE')
        # forecast_periods = self.forecast_periods
        # Calculate monthly cost - ensure the full amount is distributed
        monthly_cost = self.additional_advertisement_cost / len(forecast_periods) if len(forecast_periods) > 0 else 0

        # Set all forecast periods to 0 first (to clear any existing values)
        for period in self.forecast_periods:
            period_str = period.strftime('%b-%y')
            self.cashflow_df.loc[target_category, period_str] = 0.0

        # Distribute across forecast periods
        for period in forecast_periods:
            period_str = period.strftime('%b-%y')
            self.cashflow_df.loc[target_category, period_str] = -monthly_cost

        # Explicitly calculate and set the forecast total (Jan-25 to Exit)
        forecast_cols = [p.strftime('%b-%y') for p in forecast_periods]
        forecast_total = sum(-monthly_cost for _ in forecast_periods)

        # The name of the forecast total column
        forecast_total_col = f"{self.date_processor.start_date.strftime('%b-%y')} to Exit"

        # Set the forecast total directly
        self.cashflow_df.loc[target_category, forecast_total_col] = forecast_total

        # Set the grand total as well
        historical_total = self.cashflow_df.loc[target_category, self.date_processor.overview_columns[0]]
        self.cashflow_df.loc[target_category, 'Total'] = historical_total + forecast_total

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
        self.forecast_additional_advertisement_cost()

        # Calculate interim totals
        self.calculate_totals()

        # Calculate management fee (needs totals)
        self.forecast_management_fee()

        # Add financing costs from debt calculation
        self.populate_financing_costs()

        # Recalculate totals with management fee
        self.calculate_totals()

        # Calculate period summaries
        self.calculate_all_period_totals()

        return self.cashflow_df.round(2)