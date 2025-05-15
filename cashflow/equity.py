# cashflow/equity.py
import pandas as pd
from config import DEFAULT_CONFIG
from .base import BaseCashflowEngine


class EquityCashflow(BaseCashflowEngine):
    """Handles equity-specific cash flow calculations."""

    def __init__(
            self,
            bank_statement_df,
            acquisition_date,
            start_date,
            end_date,
            forecast_periods_count=None,
            input_assumptions=None,
    ):
        super().__init__(
            acquisition_date,
            start_date,
            end_date,
            forecast_periods_count=forecast_periods_count,
        )

        self.bank_statement_df = bank_statement_df.copy()
        self.input_assumptions = input_assumptions if input_assumptions else {}
        self.current_date = pd.Timestamp.today()
        self.cutoff_date = DEFAULT_CONFIG['cutoff_date']
        # Initialize the DataFrame with Category column
        self.equity_cashflow_df = self.initialise_dataframe(
            ['Actual/Forecast'],
            include_category_col=True
        )

    def populate_historical_equity(self):
        """Populate historical equity from bank statement data."""
        shareholders = self.get_shareholder_names()

        # Add a section header row
        equity_header = {"Category": "Shareholder Capital"}
        for col in self.cashflow_df.columns:
            if col != "Category":
                equity_header[col] = ""

        self.cashflow_df = pd.concat(
            [self.cashflow_df, pd.DataFrame([equity_header])],
            ignore_index=True
        )

        # Process each shareholder
        for shareholder_name in shareholders:
        # Process each shareholder
            # Initialize a row for this shareholder
            shareholder_row = {"Category": shareholder_name}
            for col in self.cashflow_df.columns:
                if col != "Category":
                    shareholder_row[col] = 0.0

            # Find all transactions for this shareholder in the bank statement
            for period in self.date_processor.all_periods:
                if period > self.date_processor.start_date:
                    continue  # Skip forecasting periods for now

                period_str = period.strftime('%b-%y')

                # Look for this tag in the bank statement
                period_data = self.bank_statement_df[
                    (self.bank_statement_df['TAG'] == shareholder_name) &
                    (self.bank_statement_df['MONTH'] == period)
                    ]

                # Sum the movements for this period
                if not period_data.empty:
                    amount = period_data['MOVEMENT'].sum()
                    shareholder_row[period_str] = amount

            # Add the shareholder row
            self.cashflow_df = pd.concat(
                [self.cashflow_df, pd.DataFrame([shareholder_row])],
                ignore_index=True
            )

        # Add a total row
        self.add_equity_total_row()

    @staticmethod
    def get_shareholder_names():
        return DEFAULT_CONFIG['shareholder_names']

    def add_equity_total_row(self):
        """Add a total row for all shareholder capital."""
        # Get shareholder rows
        shareholder_rows = []
        shareholders = self.get_shareholder_names()

        for idx, row in self.cashflow_df.iterrows():
            if row['Category'] in shareholders:
                shareholder_rows.append(idx)

        if not shareholder_rows:
            return

        # Create total row
        total_row = {"Category": "Total Shareholder Capital"}

        for col in self.cashflow_df.columns:
            if col == "Category":
                continue

            # Calculate total for this column
            total = 0.0
            for idx in shareholder_rows:
                val = self.cashflow_df.at[idx, col]
                if isinstance(val, (int, float)):
                    total += val

            total_row[col] = total

        # Add the total row
        self.cashflow_df = pd.concat(
            [self.cashflow_df, pd.DataFrame([total_row])],
            ignore_index=True
        )

    def forecast_equity(self):
        """Forecast equity contributions for future periods.
        This could be extended based on specific forecasting logic.
        """
        # For now, we'll assume no additional equity is contributed
        # This is a placeholder for future equity forecasting
        pass

    def add_equity_injection_total(self):
        """Add a total row for cumulative equity injection."""
        # Get shareholder rows
        shareholder_rows = []
        shareholders = self.get_shareholder_names()

        for idx, row in self.cashflow_df.iterrows():
            if row['Category'] in shareholders:
                shareholder_rows.append(idx)

        if not shareholder_rows:
            return

        # Create total injection row
        total_row = {"Category": "Total Equity Injection"}

        # Initialize running total for each column
        running_totals = {}
        for col in self.cashflow_df.columns:
            if col == "Category":
                continue
            running_totals[col] = 0.0

        # Calculate cumulative totals
        for col in self.cashflow_df.columns:
            if col == "Category":
                continue

            # Add this period's values to running total
            for idx in shareholder_rows:
                val = self.cashflow_df.at[idx, col]
                if isinstance(val, (int, float)):
                    running_totals[col] += val

            # Set the cumulative total for this column
            total_row[col] = running_totals[col]

        # Add the total injection row
        self.cashflow_df = pd.concat(
            [self.cashflow_df, pd.DataFrame([total_row])],
            ignore_index=True
        )

    def add_required_equity(self):
        """Add a row for Required Equity (Total Project Cashflow - Total Debt Cashflow)."""
        if self.project_cashflow_df is None or self.debt_cashflow_df is None:
            return

        # Create required equity row
        required_equity_row = {"Category": "Required Equity"}

        # Find Total Project Cashflow in project cashflow
        project_total = None
        if "Total Project Cashflow" in self.project_cashflow_df.index:
            project_total = self.project_cashflow_df.loc["Total Project Cashflow"]

        # Find Total Debt Cashflow in debt cashflow
        debt_total = None
        for idx, row in self.debt_cashflow_df.iterrows():
            if isinstance(row, pd.Series) and row.get("Category") == "Total Debt Cashflow":
                debt_total = row
                break

        # Calculate required equity for each column
        if project_total is not None and debt_total is not None:
            for col in self.cashflow_df.columns:
                if col == "Category":
                    continue

                # Get project cashflow value
                project_value = 0.0
                if col in project_total:
                    project_value = project_total[col]
                    if isinstance(project_value, str):
                        try:
                            project_value = float(project_value.replace('£', '').replace(',', ''))
                        except:
                            project_value = 0.0

                # Get debt cashflow value
                debt_value = 0.0
                if col in debt_total:
                    debt_value = debt_total[col]
                    if isinstance(debt_value, str):
                        try:
                            debt_value = float(debt_value.replace('£', '').replace(',', ''))
                        except:
                            debt_value = 0.0

                # Calculate required equity
                required_equity_row[col] = project_value - debt_value

        # Add the required equity row
        self.cashflow_df = pd.concat(
            [self.cashflow_df, pd.DataFrame([required_equity_row])],
            ignore_index=True
        )

    def generate_cashflow(self, project_cashflow_df=None, debt_cashflow_df=None):
        """Generate the complete equity cashflow."""
        # Reset the cashflow DataFrame
        self.cashflow_df = self.initialise_dataframe(['Actual/Forecast'], include_category_col=True)

        # Populate historical equity
        self.populate_historical_equity()

        # Forecast equity (placeholder)
        self.forecast_equity()

        # Add total equity injection row
        self.add_equity_injection_total()

        # Add required equity row if project and debt cashflows are available
        self.add_required_equity()

        print(project_cashflow_df, debt_cashflow_df)
        # Calculate Required Equity if we have the necessary data
        if project_cashflow_df is not None and debt_cashflow_df is not None:
            self.calculate_required_equity(project_cashflow_df, debt_cashflow_df)

        # Calculate period totals for each row
        numerical_rows = []
        for idx, row in self.cashflow_df.iterrows():
            if row['Category'] not in ['Actual/Forecast']:
                numerical_rows.append(idx)

        # Calculate summary columns
        for column_type in ['inception_to_cutoff', 'start_to_exit', 'total']:
            # Determine target column and date range
            if column_type == 'inception_to_cutoff':
                target_col = self.date_processor.overview_columns[0]
                start_date = self.date_processor.acquisition_date
                end_date = self.date_processor.cutoff_date
            elif column_type == 'start_to_exit':
                target_col = self.date_processor.overview_columns[1]
                start_date = self.date_processor.start_date
                end_date = self.date_processor.end_date
            elif column_type == 'total':
                target_col = self.date_processor.overview_columns[2]
                start_date = self.date_processor.acquisition_date
                end_date = self.date_processor.calculated_end_date

            # Get periods within range
            period_dates = [p for p in self.date_processor.all_periods if start_date <= p <= end_date]
            period_cols = [p.strftime('%b-%y') for p in period_dates]

            # Sum values for each numerical row
            for idx in numerical_rows:
                values = [
                    float(self.cashflow_df.at[idx, col])
                    for col in period_cols
                    if col in self.cashflow_df.columns and
                       isinstance(self.cashflow_df.at[idx, col], (int, float))
                ]
                self.cashflow_df.at[idx, target_col] = sum(values)

        return self.cashflow_df
