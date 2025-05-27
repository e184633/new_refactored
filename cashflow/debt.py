# cashflow/debt.py - COMPACT VERSION
import pandas as pd
from datetime import datetime
from config import CATEGORIES, DEFAULT_CONFIG
from .base import BaseCashflowEngine


class LoanProcessor:
    """Processes loan data for debt cashflow analysis."""

    @staticmethod
    def preprocess_loan_data(df, loan_key, date_processor):
        """Process loan dataframe for debt cashflow analysis."""
        if df is None or df.empty:
            return pd.DataFrame()

        relevant_categories = CATEGORIES.get(loan_key, [])
        if not relevant_categories:
            return pd.DataFrame()

        # Initialize result DataFrame
        columns = date_processor.overview_columns + date_processor.monthly_period_strs
        result_df = pd.DataFrame(0.0, index=relevant_categories, columns=columns)

        # Clean and process input data
        cleaned_df = df.copy()
        cleaned_df['MONTH'] = pd.to_datetime(cleaned_df['MONTH'], errors='coerce')
        cleaned_df = cleaned_df.dropna(subset=['MONTH'])
        cleaned_df['detail_2_clean'] = cleaned_df['DETAIL 2'].astype(str).str.lower().str.strip()
        cleaned_df['cash_out_adjusted'] = -pd.to_numeric(cleaned_df['CASH OUT'], errors='coerce').fillna(0.0)
        cleaned_df['month_str'] = cleaned_df['MONTH'].dt.strftime('%b-%y')

        # Map categories and aggregate data
        category_mapping = {cat.lower().strip(): cat for cat in relevant_categories}
        cleaned_df['mapped_category'] = cleaned_df['detail_2_clean'].map(category_mapping)
        mapped_data = cleaned_df.dropna(subset=['mapped_category'])

        if mapped_data.empty:
            return result_df

        # Aggregate by category and month
        aggregated = mapped_data.groupby(['mapped_category', 'month_str'])['cash_out_adjusted'].sum().reset_index()

        # Populate result DataFrame
        for _, row in aggregated.iterrows():
            category, period, amount = row['mapped_category'], row['month_str'], row['cash_out_adjusted']
            if category in result_df.index and period in result_df.columns:
                result_df.loc[category, period] = amount

        # Calculate summary columns
        LoanProcessor._calculate_summary_columns(result_df, date_processor)
        return result_df

    @staticmethod
    def _calculate_summary_columns(df, date_processor):
        """Calculate overview columns for loan data."""
        historical_cols = [col for col in date_processor.period_strs('historical') if col in df.columns]
        forecast_cols = [col for col in date_processor.period_strs('forecast') if col in df.columns]
        all_monthly_cols = [col for col in date_processor.monthly_period_strs if col in df.columns]

        for category in df.index:
            if historical_cols:
                df.loc[category, date_processor.overview_columns[0]] = df.loc[category, historical_cols].sum()
            if forecast_cols:
                df.loc[category, date_processor.overview_columns[1]] = df.loc[category, forecast_cols].sum()
            if all_monthly_cols:
                df.loc[category, date_processor.overview_columns[2]] = df.loc[category, all_monthly_cols].sum()

    @staticmethod
    def calculate_cash_payments(df, date_processor,loan_key):
        """Calculate cash payment row from loan data."""
        cash_row = {"Category": "Cash Payment"}

        # Initialize all columns to 0
        for col in date_processor.overview_columns + date_processor.monthly_period_strs:
            cash_row[col] = 0.0

        if df is None or df.empty:
            return cash_row

        # Clean the data
        cleaned_df = df.copy()
        cleaned_df['MONTH'] = pd.to_datetime(cleaned_df['MONTH'], errors='coerce')
        cleaned_df = cleaned_df.dropna(subset=['MONTH'])
        cleaned_df['CASH IN'] = pd.to_numeric(cleaned_df['CASH IN'], errors='coerce').fillna(0.0)
        cleaned_df['CASH OUT'] = pd.to_numeric(cleaned_df['CASH OUT'], errors='coerce').fillna(0.0)
        cleaned_df['detail_2_clean'] = cleaned_df['DETAIL 2'].astype(str).str.lower().str.strip()
        cleaned_df['month_str'] = cleaned_df['MONTH'].dt.strftime('%b-%y')

        # Aggregate cash in and IMS fees by month
        monthly_cash = cleaned_df.groupby('month_str')['CASH IN'].sum()
        ims_mask = cleaned_df['detail_2_clean'].str.contains('ims', na=False)
        monthly_ims_fees = cleaned_df[ims_mask].groupby('month_str')['CASH OUT'].sum()

        if loan_key == 'senior_loan':
            for period_str, amount in monthly_cash.items():
                cash_row[period_str] = amount
        else:
            # Mezzanine loan: populate all periods (dense)
            for period_str in date_processor.monthly_period_strs:
                cash_in_amount = monthly_cash.get(period_str, 0.0)
                ims_fee_amount = monthly_ims_fees.get(period_str, 0.0)
                cash_row[period_str] = cash_in_amount + ims_fee_amount
        # Calculate summary columns
        historical_cols = [col for col in date_processor.period_strs('historical')
                           if col in date_processor.monthly_period_strs]
        forecast_cols = [col for col in date_processor.period_strs('forecast')
                         if col in date_processor.monthly_period_strs]

        cash_row[date_processor.overview_columns[0]] = sum(cash_row.get(col, 0.0) for col in historical_cols)
        cash_row[date_processor.overview_columns[1]] = sum(cash_row.get(col, 0.0) for col in forecast_cols)
        cash_row[date_processor.overview_columns[2]] = sum(cash_row.get(col, 0.0)
                                                           for col in date_processor.monthly_period_strs)
        return cash_row

    @staticmethod
    def calculate_loan_redemption(section_rows, date_processor, refinancing_date):
        """Calculate loan redemption amount for refinancing date."""
        redemption_row = {"Category": "Loan Redemption"}

        # Initialize overview columns
        for col in date_processor.overview_columns:
            redemption_row[col] = ""

        # Initialize all monthly columns to 0
        for period_str in date_processor.monthly_period_strs:
            redemption_row[period_str] = 0.0

        # Calculate redemption for refinancing period
        for period_str in date_processor.monthly_period_strs:
            try:
                period_dt = datetime.strptime(period_str, '%b-%y')
                if period_dt.date() == refinancing_date.date():
                    # Calculate running balance up to this period
                    running_balance = 0.0
                    for prev_period in date_processor.monthly_period_strs:
                        if prev_period == period_str:
                            break

                        # Sum movements for previous period
                        period_total = 0.0
                        for row in section_rows:
                            if (row['Category'] not in ['Opening balance', 'Closing balance, incl rolled interest',
                                                        'Sub Total'] and
                                    prev_period in row and isinstance(row[prev_period], (int, float))):
                                period_total += row[prev_period]
                        running_balance += period_total

                    # Add current period movements
                    current_movements = 0.0
                    for row in section_rows:
                        if (row['Category'] not in ['Opening balance', 'Closing balance, incl rolled interest',
                                                    'Sub Total'] and
                                period_str in row and isinstance(row[period_str], (int, float))):
                            current_movements += row[period_str]

                    closing_balance = running_balance + current_movements
                    redemption_row[period_str] = -closing_balance if closing_balance != 0 else 0.0
                    break
            except ValueError:
                continue

        return redemption_row

    @staticmethod
    def calculate_subtotal(section_rows, date_processor):
        """Calculate subtotal row summing all loan movements."""
        subtotal_row = {"Category": "Sub Total"}

        # Initialize overview columns as empty
        for col in date_processor.overview_columns:
            subtotal_row[col] = ""

        # Calculate subtotals for each period
        for period_str in date_processor.monthly_period_strs:
            period_total = 0.0
            for row in section_rows:
                if (row['Category'] not in ['Opening balance', 'Closing balance, incl rolled interest'] and
                        period_str in row and isinstance(row[period_str], (int, float))):
                    period_total += row[period_str]
            subtotal_row[period_str] = period_total

        return subtotal_row

    @staticmethod
    def calculate_balances(section_rows, date_processor):
        """Calculate opening and closing balance rows."""
        opening_row = {"Category": "Opening balance"}
        closing_row = {"Category": "Closing balance, incl rolled interest"}

        # Initialize overview columns as empty
        for col in date_processor.overview_columns:
            opening_row[col] = ""
            closing_row[col] = ""

        # Calculate running balances
        running_balance = 0.0
        for period_str in date_processor.monthly_period_strs:
            # Opening balance for this period
            opening_row[period_str] = running_balance

            # Calculate period movements
            period_movements = 0.0
            for row in section_rows:
                if (row['Category'] not in ['Opening balance', 'Closing balance, incl rolled interest', 'Sub Total'] and
                        period_str in row and isinstance(row[period_str], (int, float))):
                    period_movements += row[period_str]

            # Update running balance
            running_balance += period_movements
            closing_row[period_str] = running_balance

        return opening_row, closing_row


class DebtCashflow(BaseCashflowEngine):
    """Handles debt-specific cash flow calculations."""

    def __init__(self, senior_loan_statement_df, mezzanine_loan_statement_df, acquisition_date, start_date, end_date,
                 annual_base_rate, forecast_periods_count, additional_unit_cost, start_of_cash_payment,
                 construction_end_date):
        super().__init__(acquisition_date, start_date, end_date, forecast_periods_count=forecast_periods_count)

        self.senior_loan_statement = senior_loan_statement_df
        self.mezzanine_loan_statement = mezzanine_loan_statement_df
        self.annual_base_rate = annual_base_rate
        self.additional_unit_cost = additional_unit_cost
        self.construction_end_date = construction_end_date
        self.start_of_cash_payment = start_of_cash_payment

        # Initialize the DataFrame with Category column
        self.cashflow_df = self.initialise_dataframe(['Actual/Forecast'], include_category_col=True)

    def populate_annual_base_rate(self):
        """Add annual base rate row to debt cashflow."""
        annual_rates = DEFAULT_CONFIG.get('annual_base_rates', {})
        base_rate_row = {'Category': 'Annual Base Rate'}

        for col in self.cashflow_df.columns:
            if col == 'Category' or col in self.date_processor.overview_columns:
                base_rate_row[col] = ""
            else:
                try:
                    period_dt = datetime.strptime(col, '%b-%y')
                    rate = annual_rates.get(period_dt, "")
                    base_rate_row[col] = f"{rate:.2%}" if isinstance(rate, float) else ""
                except ValueError:
                    base_rate_row[col] = ""

        self.cashflow_df = pd.concat([self.cashflow_df, pd.DataFrame([base_rate_row])], ignore_index=True)

    def generate_loan_section(self, loan_name, loan_df, loan_key, refinancing_date=None):
        """Generate a section for a specific loan."""
        section_rows = []

        # Add loan header
        header_row = {'Category': loan_name}
        for col in self.date_processor.overview_columns + self.date_processor.monthly_period_strs:
            header_row[col] = ""
        section_rows.append(header_row)

        # Process and add loan transactions
        loan_data = LoanProcessor.preprocess_loan_data(loan_df, loan_key, self.date_processor)
        if not loan_data.empty:
            for category in loan_data.index:
                row = {'Category': category}
                for col in loan_data.columns:
                    row[col] = loan_data.loc[category, col]
                section_rows.append(row)

        # Add cash payments
        cash_row = LoanProcessor.calculate_cash_payments(loan_df, self.date_processor, loan_key)
        section_rows.append(cash_row)

        # Add loan redemption if refinancing
        if refinancing_date:
            redemption_row = LoanProcessor.calculate_loan_redemption(section_rows[1:], self.date_processor,
                                                                     refinancing_date)
            section_rows.append(redemption_row)

        # Add subtotal
        subtotal_row = LoanProcessor.calculate_subtotal(section_rows[1:], self.date_processor)
        section_rows.append(subtotal_row)

        # Add balance rows
        opening_row, closing_row = LoanProcessor.calculate_balances(section_rows[1:], self.date_processor)
        section_rows.extend([opening_row, closing_row])

        # Apply development forecasting for mezzanine loan
        if loan_key == 'mezzanine_loan' and refinancing_date:
            self._apply_development_forecast(section_rows, refinancing_date)

        return section_rows

    def _apply_development_forecast(self, section_rows, refinancing_date):
        """Apply development cost forecasting to mezzanine loan."""
        construction_periods = pd.date_range(start=self.date_processor.start_date, end=self.construction_end_date,
                                             freq='ME')
        num_periods = len(construction_periods)
        additional_cost_per_period = self.additional_unit_cost / num_periods if num_periods > 0 else 0

        # Get development costs from project cashflow if available
        development_costs = {}
        if hasattr(self,
                   'project_cashflow_df') and self.project_cashflow_df is not None and 'Development' in self.project_cashflow_df.index:
            for col in self.project_cashflow_df.columns:
                if col in self.date_processor.monthly_period_strs:
                    development_costs[col] = self.project_cashflow_df.loc['Development', col]

        # Apply adjustments to development rows
        for row in section_rows:
            if row["Category"] == "Development":
                for col in self.date_processor.monthly_period_strs:
                    try:
                        period_dt = datetime.strptime(col, '%b-%y')
                        if refinancing_date.date() <= period_dt.date() <= self.start_of_cash_payment.date():
                            dev_value = development_costs.get(col, 0)
                            row[col] = dev_value + additional_cost_per_period
                    except ValueError:
                        continue

    def set_project_cashflow_df(self, project_cashflow_df):
        """Set the project cashflow DataFrame for development forecasting."""
        self.project_cashflow_df = project_cashflow_df

    def add_total_debt_cashflow(self):
        """Add a total row that sums up both senior and mezzanine loan subtotals."""
        subtotal_rows = [idx for idx, row in self.cashflow_df.iterrows() if row['Category'] == 'Sub Total']

        if len(subtotal_rows) < 2:
            return

        total_row = {"Category": "Total Debt Cashflow"}
        for col in self.cashflow_df.columns:
            if col == 'Category':
                continue

            total = sum(self.cashflow_df.at[idx, col] for idx in subtotal_rows
                        if isinstance(self.cashflow_df.at[idx, col], (int, float)))
            total_row[col] = total

        self.cashflow_df = pd.concat([self.cashflow_df, pd.DataFrame([total_row])], ignore_index=True)

    def generate_cashflow(self):
        """Generate the complete debt cashflow."""
        # Reset the cashflow DataFrame
        self.cashflow_df = self.initialise_dataframe(['Actual/Forecast'], include_category_col=True)

        # Add annual base rate
        self.populate_annual_base_rate()

        # Add loan sections
        for loan_name, loan_df, loan_key, refinancing_date in [
            ("OakNorth Loan", self.senior_loan_statement, "senior_loan", DEFAULT_CONFIG.get('refinancing_date')),
            ("Coutts Loan", self.mezzanine_loan_statement, "mezzanine_loan", None)
        ]:
            section_rows = self.generate_loan_section(loan_name, loan_df, loan_key, refinancing_date)
            section_df = pd.DataFrame(section_rows)
            self.cashflow_df = pd.concat([self.cashflow_df, section_df], ignore_index=True)

        # Add total debt cashflow row
        self.add_total_debt_cashflow()

        # Calculate summary columns
        self._calculate_summary_columns()

        return self.cashflow_df

    def _calculate_summary_columns(self):
        """Calculate summary columns for all numerical categories."""
        numerical_categories = [idx for idx, row in self.cashflow_df.iterrows()
                                if row['Category'] not in ['Opening balance', 'Closing balance, incl rolled interest',
                                                           'Annual Base Rate', 'Actual/Forecast']]

        for column_type in ['inception_to_cutoff', 'start_to_exit', 'total']:
            if column_type == 'inception_to_cutoff':
                target_col, start_date, end_date = (self.date_processor.overview_columns[0],
                                                    self.date_processor.acquisition_date,
                                                    self.date_processor.cutoff_date)
            elif column_type == 'start_to_exit':
                target_col, start_date, end_date = (self.date_processor.overview_columns[1],
                                                    self.date_processor.start_date, self.date_processor.end_date)
            else:  # total
                target_col, start_date, end_date = (self.date_processor.overview_columns[2],
                                                    self.date_processor.acquisition_date,
                                                    self.date_processor.calculated_end_date)

            period_cols = [p.strftime('%b-%y') for p in self.date_processor.all_periods if start_date <= p <= end_date]

            for idx in numerical_categories:
                values = [float(self.cashflow_df.at[idx, col]) for col in period_cols
                          if
                          col in self.cashflow_df.columns and isinstance(self.cashflow_df.at[idx, col], (int, float))]
                self.cashflow_df.at[idx, target_col] = sum(values)