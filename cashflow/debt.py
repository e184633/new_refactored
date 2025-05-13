# cashflow/debt.py
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
        monthly_cols = date_processor.monthly_period_strs

        # Initialize empty dataframe with categories
        result_df = pd.DataFrame(
            0.0,
            index=relevant_categories,
            columns=date_processor.overview_columns + monthly_cols
        )

        if not relevant_categories:
            return result_df

        # Standardize category names
        standardized_categories = {cat.lower(): cat for cat in relevant_categories}

        df = df.copy()
        df['MONTH'] = pd.to_datetime(df['MONTH'], errors='coerce')
        df.dropna(subset=['MONTH'], inplace=True)

        # Standardize the 'DETAIL 2' values for matching
        df['DETAIL_2_LOWER'] = df['DETAIL 2'].str.lower().str.strip()
        df_filtered = df[df['DETAIL_2_LOWER'].isin(standardized_categories.keys())].copy()

        if df_filtered.empty:
            return result_df

        # Map back to original casing
        df_filtered['DETAIL 2'] = df_filtered['DETAIL_2_LOWER'].map(standardized_categories)

        # Add month string for pivoting
        df_filtered['MONTH_STR'] = df_filtered['MONTH'].dt.strftime('%b-%y')

        # Pivot and process data
        pivoted = df_filtered.pivot_table(
            index='DETAIL 2',
            columns='MONTH_STR',
            values='CASH OUT',
            aggfunc='sum'
        ).infer_objects(False).fillna(0.0)

        # Flip sign as cash out doesn't register as cost
        pivoted = -pivoted

        # Populate result dataframe
        for category in pivoted.index:
            for col in pivoted.columns:
                if col in result_df.columns:
                    result_df.loc[category, col] = pivoted.loc[category, col]

        # Calculate overview columns
        historical_cols = date_processor.period_strs('historical')
        forecast_cols = date_processor.period_strs('forecast')

        for category in result_df.index:
            result_df.loc[category, date_processor.overview_columns[0]] = sum(
                result_df.loc[category, col] for col in historical_cols if col in result_df.columns
            )
            result_df.loc[category, date_processor.overview_columns[1]] = sum(
                result_df.loc[category, col] for col in forecast_cols if col in result_df.columns
            )
            result_df.loc[category, 'Total'] = sum(
                result_df.loc[category, col] for col in monthly_cols if col in result_df.columns
            )

        return result_df

    @staticmethod
    def calculate_cash_payments(df, date_processor):
        """Calculate cash payment row from loan data."""
        if df is None or df.empty:
            return {"Category": "Cash Payment"}

        df = df.copy()
        df['MONTH'] = pd.to_datetime(df['MONTH'], errors='coerce')
        df.dropna(subset=['MONTH'], inplace=True)
        df['MONTH_STR'] = df['MONTH'].dt.strftime('%b-%y')

        # Ensure CASH IN is numeric
        df['CASH IN'] = pd.to_numeric(df['CASH IN'], errors='coerce').fillna(0.0)

        # Get monthly totals
        monthly_totals = df.groupby('MONTH_STR')['CASH IN'].sum().to_dict()

        # Create cash row
        cash_row = {"Category": "Cash Payment"}

        # Fill monthly values
        for period_str in date_processor.monthly_period_strs:
            cash_row[period_str] = monthly_totals.get(period_str, 0.0)

        # Calculate overview columns
        historical_cols = date_processor.period_strs('historical')
        forecast_cols = date_processor.period_strs('forecast')

        cash_row[date_processor.overview_columns[0]] = sum(
            cash_row.get(col, 0.0) for col in historical_cols
        )
        cash_row[date_processor.overview_columns[1]] = sum(
            cash_row.get(col, 0.0) for col in forecast_cols
        )
        cash_row[date_processor.overview_columns[2]] = sum(
            cash_row.get(col, 0.0) for col in date_processor.monthly_period_strs
        )

        return cash_row

    @staticmethod
    def calculate_balances(loan_data, date_processor):
        """Calculate opening and closing balances for loan data."""
        opening_row = {"Category": "Opening balance"}
        closing_row = {"Category": "Closing balance, incl rolled interest"}

        # Clear overview columns
        for col in date_processor.overview_columns:
            opening_row[col] = ""
            closing_row[col] = ""

        prev_closing = 0.0
        for period_str in date_processor.monthly_period_strs:
            # Calculate movement sum for this period
            if isinstance(loan_data, pd.DataFrame):
                if period_str in loan_data.columns:
                    movement_sum = loan_data[period_str].sum()
                else:
                    movement_sum = 0.0
            else:  # Dictionary case
                movement_sum = 0.0
                for row in loan_data:
                    if isinstance(row, dict) and period_str in row:
                        val = row[period_str]
                        if isinstance(val, (int, float)):
                            movement_sum += val

            # Update balances
            opening_row[period_str] = prev_closing
            closing_row[period_str] = prev_closing + movement_sum
            prev_closing = closing_row[period_str]

        return opening_row, closing_row


class DebtCashflow(BaseCashflowEngine):
    """Handles debt-specific cash flow calculations."""

    def __init__(
            self,
            senior_loan_statement_df,
            mezzanine_loan_statement_df,
            acquisition_date,
            start_date,
            end_date,
            annual_base_rate
    ):
        super().__init__(acquisition_date, start_date, end_date)

        self.senior_loan_statement = senior_loan_statement_df
        self.mezzanine_loan_statement = mezzanine_loan_statement_df
        self.annual_base_rate = annual_base_rate

        # Initialize the DataFrame with Category column
        self.cashflow_df = self.initialise_dataframe(
            ['Actual/Forecast'],
            include_category_col=True
        )

    def populate_annual_base_rate(self):
        """Add annual base rate row to debt cashflow."""
        annual_rates = DEFAULT_CONFIG.get('annual_base_rates', {})
        base_rate_row = {'Category': 'Annual Base Rate'}

        for col in self.cashflow_df.columns:
            if col == 'Category' or col in self.date_processor.overview_columns:
                base_rate_row[col] = ""
            else:
                # Process columns that match the date format
                try:
                    period_dt = datetime.strptime(col, '%b-%y')
                    rate = annual_rates.get(period_dt, "")
                    base_rate_row[col] = f"{rate:.2%}" if isinstance(rate, float) else ""
                except ValueError:
                    base_rate_row[col] = ""

        self.cashflow_df = pd.concat(
            [self.cashflow_df, pd.DataFrame([base_rate_row])],
            ignore_index=True
        )

    def generate_loan_section(self, loan_name, loan_df, loan_key, refinancing_date=None):
        """Generate a section for a specific loan."""
        # Start with the loan name row
        loan_section = [{"Category": loan_name}]

        # Process loan data
        loan_data = LoanProcessor.preprocess_loan_data(
            loan_df, loan_key, self.date_processor
        )

        if loan_data.empty:
            return loan_section

        # Convert loan data to row format
        for category in loan_data.index:
            row = {"Category": category}
            for col in loan_data.columns:
                row[col] = loan_data.loc[category, col]
            loan_section.append(row)

        # Add cash payment row
        cash_row = LoanProcessor.calculate_cash_payments(
            loan_df, self.date_processor
        )
        loan_section.append(cash_row)

        # Handle refinancing if needed
        if refinancing_date:
            # Create a DataFrame for easier manipulation
            loan_df = pd.DataFrame(loan_section[1:])  # Skip loan name
            loan_df.set_index('Category', inplace=True)

            # Calculate balances
            opening_row, _ = LoanProcessor.calculate_balances(
                loan_df, self.date_processor
            )

            # Add redemption row for refinancing date
            redemption_row = {"Category": "Loan Redemption"}
            for col in self.date_processor.monthly_period_strs:
                try:
                    period_dt = datetime.strptime(col, '%b-%y')
                    if period_dt.date() == refinancing_date.date():
                        # Get opening balance
                        opening_bal = opening_row.get(col, 0.0)
                        # Calculate movement sum
                        movement_sum = 0.0
                        for category in loan_df.index:
                            if category != 'Cash Payment':  # Skip cash payments
                                movement_sum += loan_df.loc[category, col]

                        # Redemption is negative of (opening + movement)
                        redemption_row[col] = -(opening_bal + movement_sum)
                    else:
                        redemption_row[col] = 0.0
                except ValueError:
                    redemption_row[col] = 0.0

            # Add overview columns
            for col in self.date_processor.overview_columns:
                redemption_row[col] = ""

            loan_section.append(redemption_row)

        # Calculate subtotal
        subtotal_row = {"Category": "Sub Total"}
        for col in self.cashflow_df.columns:
            if col == 'Category':
                continue
            elif col in self.date_processor.overview_columns:
                subtotal_row[col] = ""
            else:
                # Sum values for this column
                subtotal = 0.0
                for row in loan_section[1:]:  # Skip loan name
                    if col in row and isinstance(row[col], (int, float)):
                        subtotal += row[col]
                subtotal_row[col] = subtotal

        loan_section.append(subtotal_row)

        # Add balance rows
        df_for_balances = pd.DataFrame(loan_section[1:])  # Skip loan name
        df_for_balances.set_index('Category', inplace=True)

        opening_row, closing_row = LoanProcessor.calculate_balances(
            df_for_balances, self.date_processor
        )

        loan_section.append(opening_row)
        loan_section.append(closing_row)

        return loan_section

    def generate_cashflow(self):
        """Generate the complete debt cashflow."""
        # Add annual base rate
        self.populate_annual_base_rate()

        # Add OakNorth loan section
        oaknorth_section = self.generate_loan_section(
            "OakNorth Loan",
            self.senior_loan_statement,
            "senior_loan",
            DEFAULT_CONFIG.get('refinancing_date')
        )
        self.cashflow_df = pd.concat(
            [self.cashflow_df, pd.DataFrame(oaknorth_section)],
            ignore_index=True
        )

        # Add Coutts loan section
        coutts_section = self.generate_loan_section(
            "Coutts Loan",
            self.mezzanine_loan_statement,
            "mezzanine_loan"
        )
        self.cashflow_df = pd.concat(
            [self.cashflow_df, pd.DataFrame(coutts_section)],
            ignore_index=True
        )

        # Clear overview columns for balance rows
        for idx, row in self.cashflow_df.iterrows():
            if row['Category'] in ['Opening balance', 'Closing balance, incl rolled interest']:
                for col in self.date_processor.overview_columns:
                    self.cashflow_df.at[idx, col] = ""

        # Calculate summary columns for numerical rows
        for idx, row in self.cashflow_df.iterrows():
            category = row['Category']
            if category in ['Opening balance', 'Closing balance, incl rolled interest',
                            'Annual Base Rate', 'Actual/Forecast']:
                continue

            # Calculate inception total
            inception_cols = self.date_processor.period_strs('historical')
            values = [
                float(row[col]) for col in inception_cols
                if col in self.cashflow_df.columns and isinstance(row[col], (int, float))
            ]
            self.cashflow_df.at[idx, self.date_processor.overview_columns[0]] = sum(values)

            # Calculate forecast total
            forecast_cols = self.date_processor.period_strs('forecast')
            values = [
                float(row[col]) for col in forecast_cols
                if col in self.cashflow_df.columns and isinstance(row[col], (int, float))
            ]
            self.cashflow_df.at[idx, self.date_processor.overview_columns[1]] = sum(values)

            # Calculate grand total
            monthly_cols = self.date_processor.monthly_period_strs
            values = [
                float(row[col]) for col in monthly_cols
                if col in self.cashflow_df.columns and isinstance(row[col], (int, float))
            ]
            self.cashflow_df.at[idx, self.date_processor.overview_columns[2]] = sum(values)

        return self.cashflow_df
    