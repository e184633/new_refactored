import pandas as pd


def parse_date(date_str: str, date_format: str = "%b-%y") -> pd.Timestamp:
    """ Parse a date string into a datetime object. """
    try:
        return pd.to_datetime(date_str, format=date_format)
    except ValueError as e:
        raise ValueError(f"Invalid date format for '{date_str}'. Expected format: MMM-YY (e.g., Jan-25).") from e


def format_cf_value(value):
    """Format a value for cashflow table."""
    # If value is a Series, extract the first element (or handle as needed)
    if isinstance(value, pd.Series):
        value = value.iloc[0]

    if pd.isna(value) or value == "-" or value == "":
        return "-"

    if isinstance(value, str) and value.endswith("%"):
        return value

    try:
        value = float(value)
        if value == 0:
            return "-"
        elif value < 0:
            return f"({abs(value):,.0f})"
        else:
            return f"{value:,.0f}"
    except (ValueError, TypeError):
        return str(value) if not pd.isna(value) else "-"
