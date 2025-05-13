# cashflow/base.py
from datetime import datetime

import pandas as pd

from config import DEFAULT_CONFIG


class DatePeriodProcessor:
    """Handles all date processing and period calculations."""

    def __init__(self, acquisition_date, start_date, end_date, cutoff_date=None):
        """Initialize date processor with key project dates."""
        self.acquisition_date = pd.Timestamp(acquisition_date)
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.cutoff_date = (pd.Timestamp(cutoff_date) if cutoff_date
                            else pd.Timestamp(DEFAULT_CONFIG['cutoff_date']))

        # Generate period ranges
        self.all_periods = pd.date_range(
            start=self.acquisition_date,
            end=self.end_date,
            freq='ME'
        )
        self.monthly_period_strs = [p.strftime('%b-%y') for p in self.all_periods]

        # Define overview columns
        self.overview_columns = [
            f"Inception to {self.cutoff_date.strftime('%b-%y')}",
            f"{self.start_date.strftime('%b-%y')} to Exit",
            'Total'
        ]

    def filter_periods(self, period_type):
        """Filter periods based on type (historical, forecast, all)."""
        if period_type == 'historical':
            return [p for p in self.all_periods if p <= self.cutoff_date]
        elif period_type == 'forecast':
            return [p for p in self.all_periods if p >= self.start_date]
        return self.all_periods

    def period_strs(self, period_type='all'):
        """Get period strings for the given type."""
        periods = self.filter_periods(period_type)
        return [p.strftime('%b-%y') for p in periods]

    def calculate_period_totals(self, df, period_type, numerical_categories=None):
        """Calculate totals for a specific period type."""
        # Determine target column based on period type
        if period_type == 'historical':
            total_col = self.overview_columns[0]
        elif period_type == 'forecast':
            total_col = self.overview_columns[1]
        else:  # 'all'
            total_col = self.overview_columns[2]

        # Get period columns
        period_cols = self.period_strs(period_type)

        # Determine categories to calculate
        if numerical_categories is None:
            numerical_categories = [cat for cat in df.index if cat != 'Actual/Forecast']

        # Calculate sums for each numerical category
        df.loc[numerical_categories, total_col] = df.loc[numerical_categories, period_cols].sum(axis=1)

        # Handle 'Actual/Forecast' row if present
        if 'Actual/Forecast' in df.index:
            if period_type == 'historical':
                df.loc['Actual/Forecast', total_col] = 'Actual'
            elif period_type == 'forecast':
                df.loc['Actual/Forecast', total_col] = 'Forecast'
            else:  # 'all'
                df.loc['Actual/Forecast', total_col] = 'Actual/Forecast'

        return df


class BaseCashflowEngine:
    """Base class with common cashflow functionality."""

    def __init__(
            self,
            acquisition_date,
            start_date,
            end_date,
            cutoff_date=None
    ):
        self.date_processor = DatePeriodProcessor(
            acquisition_date, start_date, end_date, cutoff_date
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
                            col] = 'Actual' if period_dt < self.date_processor.start_date else 'Forecast'

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
                df.loc['Actual/Forecast', self.date_processor.overview_columns[0]] = 'Actual'
                df.loc['Actual/Forecast', self.date_processor.overview_columns[1]] = 'Forecast'
                df.loc['Actual/Forecast', 'Total'] = 'Actual/Forecast'

                for period in self.date_processor.all_periods:
                    period_str = period.strftime('%b-%y')
                    df.loc[
                        'Actual/Forecast', period_str] = 'Actual' if period < self.date_processor.start_date else 'Forecast'

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

    def generate_cashflow(self):
        """Generate cashflow - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")
