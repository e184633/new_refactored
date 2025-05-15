# cashflow/base.py
from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta

from config import DEFAULT_CONFIG


class DatePeriodProcessor:
    """Handles all date processing and period calculations."""

    def __init__(self, acquisition_date, start_date, end_date, forecast_periods_count=None):
        """Initialize date processor with key project dates."""
        self.acquisition_date = pd.Timestamp(acquisition_date)
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date) + relativedelta(days=31)
        self.cutoff_date = pd.Timestamp(DEFAULT_CONFIG['cutoff_date']) + relativedelta(days=30)
        self.current_date = pd.Timestamp.today()
        # Calculate the actual end date based on start_date + forecast_periods_count
        if forecast_periods_count is not None:
            self.calculated_end_date = pd.Timestamp(end_date) + pd.DateOffset(months=forecast_periods_count)
        else:
            self.calculated_end_date = pd.Timestamp(end_date) if end_date else None

        # Generate period ranges
        self.all_periods = pd.date_range(
            start=self.acquisition_date,
            end=self.calculated_end_date,
            freq='ME'
        )
        self.monthly_period_strs = [p.strftime('%b-%y') for p in self.all_periods]

        # Define overview columns
        self.overview_columns = [
            f"Inception to {self.cutoff_date.strftime('%b-%y')}",
            f"{self.start_date.strftime('%b-%y')} to Exit",
            'Total'
        ]
        # # But historical/forecast split is based on current date
        # self.historical_periods = [p for p in self.all_periods if p <= self.current_date]
        # self.forecast_periods = [p for p in self.all_periods if p > self.current_date]

    def filter_periods(self, period_type):
        """Filter periods based on type (historical, forecast, all)."""
        if period_type == 'historical':
            return [p for p in self.all_periods if p <= self.current_date]
        elif period_type == 'forecast':
            return [p for p in self.all_periods if p >= self.current_date]
        elif period_type == 'start_to_exit':
            return [p for p in self.all_periods if p >= self.start_date]
        return self.all_periods

    # In cashflow/base.py, add this method to DatePeriodProcessor class:
    def get_periods_until_exit(self, periods=None):
        """Get periods until the exit date.

        Args:
            periods: Periods to filter (defaults to all_periods if None)

        Returns:
            List of periods that occur on or before the exit date
        """
        if periods is None:
            periods = self.all_periods

        return [p for p in periods if p <= self.end_date]

    def filter_periods_until_date(self, periods, end_date=None):
        """Filter periods to include only those up to the specified end date.

        Args:
            periods: List of periods to filter
            end_date: End date cutoff (defaults to self.end_date if None)

        Returns:
            List of periods that are <= end_date
        """
        if end_date is None:
            end_date = self.end_date

        return [p for p in periods if p <= end_date]

    def period_strs(self, period_type='all'):
        """Get period strings for the given type."""
        periods = self.filter_periods(period_type)
        return [p.strftime('%b-%y') for p in periods]

    def calculate_period_totals(self, df, column_type, numerical_categories=None):
        """Calculate totals for a specific summary column."""
        # Determine which columns to sum and which summary column to update
        if column_type == 'inception_to_cutoff':
            # "Inception to Dec-24" column
            target_col = self.overview_columns[0]
            start_date = self.acquisition_date
            end_date = self.cutoff_date
        elif column_type == 'start_to_exit':
            # "Jan-25 to Exit" column
            target_col = self.overview_columns[1]
            start_date = self.start_date
            end_date = self.calculated_end_date
        elif column_type == 'total':
            # "Total" column
            target_col = self.overview_columns[2]
            start_date = self.acquisition_date
            end_date = self.calculated_end_date
        else:
            raise ValueError(f"Unknown column_type: {column_type}")

        # Get monthly columns within the date range
        period_dates = [p for p in self.all_periods if start_date <= p <= end_date]
        period_cols = [p.strftime('%b-%y') for p in period_dates]

        # Handle empty period list
        if not period_cols:
            return df

        # Determine categories to calculate
        if numerical_categories is None:
            numerical_categories = [cat for cat in df.index if cat != 'Actual/Forecast']

        # Calculate sums for numerical categories
        for category in numerical_categories:
            # Filter valid columns that exist in the dataframe
            valid_cols = [col for col in period_cols if col in df.columns]
            if valid_cols:
                # Fix: Remove axis=1 since df.loc[category, valid_cols] returns a Series
                df.loc[category, target_col] = df.loc[category, valid_cols].sum()

        # Set the text for the 'Actual/Forecast' row if present
        if 'Actual/Forecast' in df.index:
            if column_type == 'inception_to_cutoff':
                df.loc['Actual/Forecast', target_col] = 'Actual'
            elif column_type == 'start_to_exit':
                df.loc['Actual/Forecast', target_col] = 'Forecast'
            elif column_type == 'total':
                df.loc['Actual/Forecast', target_col] = 'Actual/Forecast'

        return df

class BaseCashflowEngine:
    """Base class with common cashflow functionality."""

    def __init__(
            self,
            acquisition_date,
            start_date,
            end_date,
            forecast_periods_count=None,
    ):
        self.date_processor = DatePeriodProcessor(
            acquisition_date, start_date, end_date, forecast_periods_count
        )
        self.cashflow_df = None

    def initialise_dataframe(self, categories, include_category_col=False):
        """Initialize a dataframe with standard columns and categories."""
        # Determine columns
        if include_category_col:
            columns = ['Category'] + self.date_processor.overview_columns + self.date_processor.monthly_period_strs
        else:
            columns = self.date_processor.overview_columns + self.date_processor.monthly_period_strs

        # Create dataframe
        if include_category_col:
            df = pd.DataFrame(columns=columns)
            # Add 'Actual/Forecast' row if needed
            if 'Actual/Forecast' in categories:
                actual_forecast_row = {'Category': 'Actual/Forecast'}
                for col in columns[1:]:  # Skip 'Category'
                    if col == self.date_processor.overview_columns[0]:
                        actual_forecast_row[col] = 'Actual'
                    elif col == self.date_processor.overview_columns[1]:
                        actual_forecast_row[col] = 'Forecast'
                    elif col == self.date_processor.overview_columns[2]:
                        actual_forecast_row[col] = 'Actual/Forecast'
                    else:  # Monthly periods
                        period_dt = datetime.strptime(col, '%b-%y')
                        actual_forecast_row[
                            col] = 'Actual' if period_dt < self.date_processor.current_date else 'Forecast'

                df = pd.concat([df, pd.DataFrame([actual_forecast_row])], ignore_index=True)
                # Create remaining rows
                for category in [c for c in categories if c != 'Actual/Forecast']:
                    row = {'Category': category}
                    for col in columns[1:]:
                        row[col] = 0.0
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame(index=categories, columns=columns, dtype=object)

            # Set Actual/Forecast row
            if 'Actual/Forecast' in categories:
                for period in self.date_processor.all_periods:
                    period_str = period.strftime('%b-%y')
                    df.loc[
                        'Actual/Forecast', period_str] = 'Actual' if period <= self.date_processor.current_date else 'Forecast'

            # Initialize numerical cells to 0.0
            numerical_rows = [cat for cat in categories if cat != 'Actual/Forecast']
            df.loc[numerical_rows] = df.loc[numerical_rows].infer_objects(copy=False).fillna(0.0)

        return df

    def calculate_all_period_totals(self, df=None):
        """Calculate totals for all period types."""
        if df is None:
            df = self.cashflow_df

        numerical_categories = [cat for cat in df.index if cat != 'Actual/Forecast'] if not df.empty else []

        # Calculate for historical, forecast, and all periods
        df = self.date_processor.calculate_period_totals(
            df, 'historical', numerical_categories
        )
        df = self.date_processor.calculate_period_totals(
            df, 'forecast', numerical_categories
        )
        df = self.date_processor.calculate_period_totals(
            df, 'all', numerical_categories
        )

        return df

    def calculate_all_period_totals(self, df=None):
        """Calculate totals for all summary columns."""
        if df is None:
            df = self.cashflow_df

        numerical_categories = [cat for cat in df.index if cat != 'Actual/Forecast'] if not df.empty else []

        # Use column identifiers that reflect their purpose, not their content type
        df = self.date_processor.calculate_period_totals(
            df, 'inception_to_cutoff', numerical_categories  # First summary column
        )
        df = self.date_processor.calculate_period_totals(
            df, 'start_to_exit', numerical_categories  # Second summary column
        )
        df = self.date_processor.calculate_period_totals(
            df, 'total', numerical_categories  # Third summary column
        )

        return df

    def generate_cashflow(self):
        """Generate cashflow - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")
