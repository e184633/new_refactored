import pandas as pd
import streamlit as st
from config import CATEGORIES, DEFAULT_CONFIG
from datetime import datetime
from dateutil.relativedelta import relativedelta
from utils import format_cf_value

import plotly.express as px
SAV_DATE_FORMAT = '%b-%y'
FORECAST_START_DATE = DEFAULT_CONFIG['cutoff_date'] + relativedelta(months=1)
OVERVIEW_COLUMNS = [f"Inception to {DEFAULT_CONFIG['cutoff_date'].strftime(SAV_DATE_FORMAT)}",
                    f"{DEFAULT_CONFIG['start_date'].strftime(SAV_DATE_FORMAT)} to Exit", 'Total']
# Make sure these are correctly defined in your config.py or passed appropriately
ACQUISITION_DATE = DEFAULT_CONFIG['acquisition_date']
START_DATE = DEFAULT_CONFIG.get('start_date')  # Forecast start date
CUTOFF_DATE = DEFAULT_CONFIG.get('cutoff_date') + relativedelta(day=31)
END_DATE = DEFAULT_CONFIG.get('exit_date') + relativedelta(day=31)


def get_all_periods(acquisition_date, start_date, end_date, forecast_periods_count):
    """Determines the full range of monthly periods needed."""
    # Ensure end_date covers the forecast horizon
    forecast_end_date = start_date + relativedelta(months=forecast_periods_count - 1)
    actual_end_date = max(pd.Timestamp(end_date), forecast_end_date)
    # Ensure start date is not after end date
    actual_start_date = min(pd.Timestamp(acquisition_date), start_date)

    if actual_start_date > actual_end_date:
        # Handle case where acquisition/start is after calculated end date (e.g., short forecast)
        # This might happen if only forecast periods define the range.
        print(
            f"Warning: Acquisition/Start date ({actual_start_date}) is after calculated end date ({actual_end_date}). Adjusting period range.")
        # If start date is the driver and is after end date, maybe just use forecast range?
        if start_date > actual_end_date:
            print("Using forecast periods for date range.")
            return pd.date_range(start=start_date, periods=forecast_periods_count, freq='ME')
        else:  # Fallback if acquisition date caused the issue but end_date is valid
            return pd.date_range(start=actual_start_date, end=actual_end_date, freq='ME')

    return pd.date_range(start=actual_start_date, end=actual_end_date, freq='ME')

def preprocess_loan_df(df: pd.DataFrame, loan_key: str, all_periods: pd.DatetimeIndex, overview_columns: list,
                       start_date: datetime, cutoff_date: datetime, value_col: str = 'CASH OUT') -> pd.DataFrame:
    relevant_categories = CATEGORIES.get(loan_key, [])
    monthly_cols_str = [p.strftime(SAV_DATE_FORMAT) for p in all_periods]
    target_columns = overview_columns + monthly_cols_str

    # Standardize category names to lowercase for matching
    standardized_categories = {cat.lower(): cat for cat in relevant_categories}  # map lowercase -> proper case

    # Initialize empty dataframe
    empty_df = pd.DataFrame(0.0, index=relevant_categories, columns=target_columns)
    if df is None or df.empty or not relevant_categories:
        print(f"Warning: Empty DataFrame or no relevant categories for {loan_key}.")
        return empty_df

    df = df.copy()

    if 'MONTH' not in df.columns or 'DETAIL 2' not in df.columns or value_col not in df.columns:
        print(f"Error: Missing required columns ('MONTH', 'DETAIL 2', '{value_col}') in DataFrame for {loan_key}.")
        return empty_df

    df['MONTH'] = pd.to_datetime(df['MONTH'], errors='coerce')
    df.dropna(subset=['MONTH'], inplace=True)

    # Standardize the 'DETAIL 2' values for matching
    df['DETAIL_2_LOWER'] = df['DETAIL 2'].str.lower().str.strip()
    df_filtered = df[df['DETAIL_2_LOWER'].isin(standardized_categories.keys())].copy()

    if df_filtered.empty:
        print(f"Warning: No data found for relevant categories in {loan_key}.")
        return empty_df

    # Map back to original casing
    df_filtered['DETAIL 2'] = df_filtered['DETAIL_2_LOWER'].map(standardized_categories)

    # Add month string for pivoting
    df_filtered['MONTH_STR'] = df_filtered['MONTH'].dt.strftime(SAV_DATE_FORMAT)

    pivoted = df_filtered.pivot_table(
        index='DETAIL 2',
        columns='MONTH_STR',
        values=value_col,
        aggfunc='sum',
        fill_value=0.0
    ).infer_objects(copy=False)
    # print(pivoted.dtypes)

    # Flip sign as cash out doesn't register as cost
    pivoted = -pivoted.reindex(columns=monthly_cols_str, fill_value=0.0).infer_objects(copy=False)


    inception_periods_dt = all_periods[all_periods <= cutoff_date]
    forecast_periods_dt = all_periods[all_periods >= start_date]
    inception_cols_str = [p.strftime(SAV_DATE_FORMAT) for p in inception_periods_dt]
    forecast_cols_str = [p.strftime(SAV_DATE_FORMAT) for p in forecast_periods_dt]
    summary_df = pd.DataFrame(index=pivoted.index)
    summary_df[overview_columns[0]] = pivoted.loc[:, pivoted.columns.isin(inception_cols_str)].sum(axis=1)
    summary_df[overview_columns[1]] = pivoted.loc[:, pivoted.columns.isin(forecast_cols_str)].sum(axis=1)
    summary_df[overview_columns[2]] = pivoted.sum(axis=1)

    result_df = pd.concat([summary_df, pivoted], axis=1)
    result_df = result_df.reindex(index=relevant_categories, fill_value=0.0).infer_objects(copy=False)
    result_df = result_df.reindex(columns=target_columns, fill_value=0.0).infer_objects(copy=False)
    return result_df

def calculate_cash_payments(df: pd.DataFrame, all_periods: pd.DatetimeIndex) -> dict:
    """Aggregate CASH IN values across all categories for each period."""
    if df is None or df.empty:
        return {"Category": "Cash Payment", **{p.strftime(SAV_DATE_FORMAT): 0.0 for p in all_periods}}

    df = df.copy()
    df['MONTH'] = pd.to_datetime(df['MONTH'], errors='coerce')
    df.dropna(subset=['MONTH'], inplace=True)

    df['MONTH_STR'] = df['MONTH'].dt.strftime(SAV_DATE_FORMAT)

    # Ensure CASH IN is numeric
    df['CASH IN'] = pd.to_numeric(df['CASH IN'], errors='coerce').infer_objects(copy=False).fillna(0.0)


    monthly_totals = df.groupby('MONTH_STR')['CASH IN'].sum().to_dict()

    cash_row = {"Category": "Cash Payment"}
    for period in all_periods:
        col = period.strftime(SAV_DATE_FORMAT)
        cash_row[col] = monthly_totals.get(col, 0.0)

    # Add overview columns (Inception, Forecast, Total)
    inception_cols = [p.strftime(SAV_DATE_FORMAT) for p in all_periods if p <= CUTOFF_DATE]
    forecast_cols = [p.strftime(SAV_DATE_FORMAT) for p in all_periods if p >= START_DATE]

    cash_row[OVERVIEW_COLUMNS[0]] = sum(cash_row.get(col, 0.0) for col in inception_cols)
    cash_row[OVERVIEW_COLUMNS[1]] = sum(cash_row.get(col, 0.0) for col in forecast_cols)
    cash_row[OVERVIEW_COLUMNS[2]] = sum(
        cash_row.get(col, 0.0) for col in [p.strftime(SAV_DATE_FORMAT) for p in all_periods])

    return cash_row


def create_debt_cashflow_table(senior_loan_df, mezzanine_loan_df):
    """Create a debt cashflow breakdown table with time-series columns."""
    from config import DEFAULT_CONFIG

    # --- Configuration & Periods ---
    refinancing_date = DEFAULT_CONFIG.get('refinancing_date')
    all_periods = get_all_periods(
        ACQUISITION_DATE, START_DATE, END_DATE,
        DEFAULT_CONFIG.get('forecast_periods_count', 4)
    )
    period_cols = [p.strftime(SAV_DATE_FORMAT) for p in all_periods]
    target_columns = OVERVIEW_COLUMNS + period_cols
    annual_rates = DEFAULT_CONFIG.get('annual_base_rates', {})

    def update_or_append_row(df, new_row, category_col="Category"):
        df = df.copy()
        new_row_df = pd.DataFrame([new_row]).set_index(category_col)

        for col in new_row_df.columns:
            if col not in df.columns:
                df[col] = 0.0
        for col in df.columns:
            if col not in new_row_df.columns:
                new_row_df[col] = 0.0

        new_row_df = new_row_df[df.columns]

        if new_row_df.index[0] in df.index:
            df.loc[new_row_df.index[0]] = new_row_df.iloc[0]
        else:
            df = pd.concat([df, new_row_df])

        return df

    rows = []

    # --- Annual Base Rate Row ---
    rate_row = {"Category": "Annual Base Rate"}
    for col in target_columns:
        if col in OVERVIEW_COLUMNS:
            rate_row[col] = ""
        else:
            try:
                period_dt = datetime.strptime(col, SAV_DATE_FORMAT)
                rate = annual_rates.get(period_dt, "")
                rate_row[col] = f"{rate:.2%}" if isinstance(rate, float) else ""
            except ValueError:
                rate_row[col] = ""
    rows.append(rate_row)

    # === OakNorth Loan Section ===
    rows.append({"Category": "OakNorth Loan", **{col: "" for col in target_columns}})

    senior_data = preprocess_loan_df(
        senior_loan_df, 'senior_loan', all_periods,
        OVERVIEW_COLUMNS, START_DATE, CUTOFF_DATE, value_col='CASH OUT'
    )

    senior_cash_row = calculate_cash_payments(senior_loan_df, all_periods)
    senior_cash_row.update({
        col: sum(senior_cash_row.get(c, 0.0) for c in period_cols)
        for col in OVERVIEW_COLUMNS
    })

    senior_data = update_or_append_row(senior_data, senior_cash_row)

    updated_senior_data = senior_data.copy()
    oak_opening, _ = calculate_balances(updated_senior_data, target_columns)

    def calculate_redemption_row(opening_row):
        categories_to_sum = [
            'Acquisition', 'Fees', 'Development', 'Capitalised Interest',
            'Non-utilisation Fee', 'IMS Fees', 'Cash Payment'
        ]
        redemption_row = {"Category": "Loan Redemption"}
        for col in target_columns:
            if col in OVERVIEW_COLUMNS:
                continue
            try:
                period_dt = datetime.strptime(col, SAV_DATE_FORMAT)
            except ValueError:
                redemption_row[col] = ""
                continue
            if period_dt.date() == refinancing_date.date():
                subtotal = sum([
                    updated_senior_data.loc[cat, col]
                    for cat in categories_to_sum
                    if cat in updated_senior_data.index
                ])
                opening_val = opening_row.get(col, 0.0)
                redemption_row[col] = -(opening_val + subtotal)
            else:
                redemption_row[col] = 0.0

        # redemption_row.update({
        #     col: sum(redemption_row.get(c, 0.0) for c in period_cols)
        #     for col in OVERVIEW_COLUMNS
        # })
        redemption_row.update({
            OVERVIEW_COLUMNS[0]: sum(redemption_row.get(c, 0.0) for c in period_cols
                                     if c in redemption_row and not (
                        isinstance(redemption_row[c], str) and redemption_row[c].strip() == "")
                                     and datetime.strptime(c, SAV_DATE_FORMAT) <= CUTOFF_DATE),
            # Only include periods before cutoff
            OVERVIEW_COLUMNS[1]: 0.0,  # Force forecast period to be 0 for redemption
            OVERVIEW_COLUMNS[2]: sum(redemption_row.get(c, 0.0) for c in period_cols
                                     if c in redemption_row and not (
                        isinstance(redemption_row[c], str) and redemption_row[c].strip() == ""))
        })
        return redemption_row

    loan_redemption_row = calculate_redemption_row(oak_opening)
    updated_senior_data = update_or_append_row(updated_senior_data, loan_redemption_row)

    oak_subtotal_row = {
        "Category": "Sub Total",
        **{col: updated_senior_data[col].sum() for col in target_columns}
    }


    oak_opening, oak_closing = calculate_balances(updated_senior_data, target_columns)
    updated_senior_data = pd.concat([
        updated_senior_data,
        pd.DataFrame([oak_subtotal_row]).set_index('Category')
    ])
    for category, row in updated_senior_data.iterrows():
        rows.append({"Category": category, **row.to_dict()})

    rows.extend([oak_opening, oak_closing])

    # === Coutts Loan Section ===
    rows.append({"Category": "Coutts Loan", **{col: "" for col in target_columns}})

    mezzanine_data = preprocess_loan_df(
        mezzanine_loan_df, 'mezzanine_loan', all_periods,
        OVERVIEW_COLUMNS, START_DATE, CUTOFF_DATE, value_col='CASH OUT'
    )

    mezzanine_cash_row = calculate_cash_payments(mezzanine_loan_df, all_periods)
    mezzanine_cash_row.update({
        col: sum(mezzanine_cash_row.get(c, 0.0) for c in period_cols)
        for col in OVERVIEW_COLUMNS
    })
    mezzanine_data = update_or_append_row(mezzanine_data, mezzanine_cash_row)



    coutts_opening, coutts_closing = calculate_balances(mezzanine_data, target_columns)
    mezzanine_subtotal_row = {
        "Category": "Sub Total",
        **{col: mezzanine_data[col].sum() for col in target_columns}
    }
    mezzanine_data = pd.concat([
        mezzanine_data,
        pd.DataFrame([mezzanine_subtotal_row]).set_index('Category')
    ])
    for category, row in mezzanine_data.iterrows():
        rows.append({"Category": category, **row.to_dict()})

    rows.extend([coutts_opening, coutts_closing])

    final_columns_order = ['Category'] + target_columns
    return pd.DataFrame(rows)[final_columns_order]


def display_debt_cashflow(debt_df):
    """Display the debt cashflow table with consistent styling and no checkmarks."""
    st.subheader("Financing Cashflow")
    if debt_df.empty:
        st.warning("No data available to display in Debt Cashflow table.")
        return

    col_headers = debt_df.columns.tolist()

    # Style constants
    header_style = 'border: 1px solid #ddd; padding: 8px; min-width: 120px;'
    sticky_style = 'position: sticky; z-index: 1;'
    column_header_style = f'{header_style} text-align: right;'
    row_style = 'border: 1px solid #ddd; padding: 8px; min-width: 120px;'

    # Define columns to freeze
    frozen_columns = OVERVIEW_COLUMNS[:3] if OVERVIEW_COLUMNS else col_headers[:3]
    column_width = 119
    left_positions = {col: (i + 1) * column_width for i, col in enumerate(frozen_columns)}

    # Categories for styling
    header_categories = ["OakNorth Loan", "Coutts Loan", "Annual Base Rate"]
    subtotal_categories = ["Sub Total", "Sub Total"]
    bold_categories = header_categories + subtotal_categories

    # Build HTML
    html = '<div style="overflow-x: auto; max-width: 100%;">'
    html += '<table style="width:100%; border-collapse: collapse; margin: 0;">'

    # Header row
    html += '<tr style="background-color: #f2f2f2;">'
    category_header_style = f'{header_style} text-align: left; background-color: #f2f2f2; {sticky_style} left: 0px;'
    html += f'<th style="{category_header_style}">Category</th>'
    for col in col_headers[1:]:
        if col in frozen_columns:
            left_pos = left_positions[col]
            col_header_style = f'{header_style} text-align: right; background-color: #f2f2f2; {sticky_style} left: {left_pos}px;'
            html += f'<th style="{col_header_style}">{col}</th>'
        else:
            html += f'<th style="{column_header_style}">{col}</th>'
    html += '</tr>'

    # Data rows
    for _, row in debt_df.iterrows():
        category = row["Category"]
        is_main_category = category in bold_categories
        bg_color = "#e6f3ff" if category in header_categories else "#ffffff"
        font_weight = "bold" if is_main_category else "normal"

        html += f'<tr style="background-color: {bg_color};">'
        category_style = f'{row_style} font-weight: {font_weight}; white-space: pre; {sticky_style} background-color: {bg_color}; left: 0px;'
        html += f'<td style="{category_style}">{category}</td>'
        for col in col_headers[1:]:
            val = format_cf_value(row[col])
            text_align = 'right'
            if col in frozen_columns:
                left_pos = left_positions[col]
                cell_style = f'{row_style} text-align: {text_align}; background-color: {bg_color}; {sticky_style} left: {left_pos}px;'
                html += f'<td style="{cell_style}">{val}</td>'
            else:
                cell_style = f'{row_style} text-align: {text_align};'
                html += f'<td style="{cell_style}">{val}</td>'
        html += '</tr>'

    html += '</table>'
    html += '</div>'

    # Render
    st.markdown(html, unsafe_allow_html=True)


def create_debt_charts(debt_df):
    """
    Visualize debt cashflows per loan and category.
    Assumes 'Category' column and 'Total' column exist.
    Automatically detects loan headers like "OakNorth Loan" or "Coutts Loan".
    """
    st.subheader("Debt Visualization (Absolute Totals, Log Scale)")

    if debt_df.empty:
        st.warning("No data available for charts.")
        return

    if "Category" not in debt_df.columns or "Total" not in debt_df.columns:
        st.error("DataFrame must contain 'Category' and 'Total' columns.")
        return

    # Clean numeric values
    def clean_numeric(val):
        if isinstance(val, (int, float)):
            return val
        if isinstance(val, str):
            val = val.replace(',', '').replace('(', '-').replace(')', '').replace('%', '')
        try:
            return float(val)
        except:
            return 0.0

    df = debt_df.copy()
    df["Total"] = df["Total"].apply(clean_numeric)
    df["AbsAmount"] = df["Total"].abs()

    # Identify current loan as we go down the rows
    current_loan = None
    loan_rows = []
    for _, row in df.iterrows():
        category = row["Category"]
        if category == 'Loan Redemption':
            continue
        if "Loan" in category:
            current_loan = category.replace(" Loan", "")
        elif category in ["Opening balance", "Closing balance, incl rolled interest", "Annual Base Rate"]:
            continue  # skip non-data rows
        elif pd.notnull(current_loan):
            loan_rows.append({
                "Loan": current_loan,
                "Category": category,
                "Total": row["Total"],
                "AbsAmount": row["AbsAmount"]
            })

    # Convert to a clean DataFrame
    chart_df = pd.DataFrame(loan_rows)
    if chart_df.empty:
        st.warning("No valid loan data found.")
        return

    category_colors = px.colors.qualitative.Set3
    loan_colors = {"OakNorth": "#1F77B4", "Coutts": "#FF7F0E"}

    for loan in chart_df["Loan"].unique():
        loan_data = chart_df[chart_df["Loan"] == loan]
        st.subheader(f"{loan} | Absolute Cash Flow by Category")
        fig = px.bar(
            loan_data,
            x="Category",
            y="AbsAmount",
            color="Category",
            title=f"{loan} Loan - Absolute Cash Flow (Log Scale)",
            color_discrete_sequence=category_colors,
            labels={"AbsAmount": "Absolute Amount (£)", "Category": "Category"}
        )
        fig.update_layout(xaxis_tickangle=-45)
        fig.update_yaxes(type='log')
        st.plotly_chart(fig, use_container_width=False)

    # Combined bar chart
    st.subheader("Combined Loan Cash Flow (Grouped by Loan, Log Scale)")
    fig_combined = px.bar(
        chart_df,
        x="Category",
        y="AbsAmount",
        color="Loan",
        barmode="group",
        color_discrete_map=loan_colors,
        title="Combined | Absolute Cash Flow (Log Scale)",
        labels={"AbsAmount": "Absolute Amount (£)", "Category": "Category", "Loan": "Loan Type"}
    )
    fig_combined.update_layout(xaxis_tickangle=-45)
    fig_combined.update_yaxes(type='log')
    st.plotly_chart(fig_combined, use_container_width=False)

    # Pie chart
    st.subheader("Loan Composition by Absolute Value")
    pie_df = chart_df.groupby("Loan")["AbsAmount"].sum().reset_index()
    fig_pie = px.pie(
        pie_df,
        values="AbsAmount",
        names="Loan",
        title="Loan Composition (Total Absolute Value)",
        color_discrete_map=loan_colors
    )
    st.plotly_chart(fig_pie, use_container_width=True)


def calculate_balances(cashflow_df, target_columns):
    """
        Returns monthly opening and closing balances based on cash movements.
        Assumes positive = disbursement / rolled interest, negative = repayment.
        """
    opening_row = {"Category": "Opening balance"}
    closing_row = {"Category": "Closing balance, incl rolled interest"}

    monthly_cols = [col for col in target_columns if col not in OVERVIEW_COLUMNS]
    prev_closing = 0.0

    for col in monthly_cols:
        movement_sum = cashflow_df[col].sum()
        opening_row[col] = prev_closing
        closing_row[col] = prev_closing + movement_sum
        prev_closing = closing_row[col]

    # for col in OVERVIEW_COLUMNS:
    #     opening_row[col] = sum(opening_row.get(mc, 0.0) for mc in monthly_cols if mc in opening_row)
    #     closing_row[col] = sum(closing_row.get(mc, 0.0) for mc in monthly_cols if mc in closing_row)
    return opening_row, closing_row


def add_debt_cashflow_tab(senior_loan_df, mezzanine_loan_df):
    """Add the debt cashflow tab to the dashboard."""

    debt_df = create_debt_cashflow_table(senior_loan_df, mezzanine_loan_df)
    display_debt_cashflow(debt_df)

    # --- Add visualization charts ---
    if not debt_df.empty:
        try:
            create_debt_charts(debt_df)
        except Exception as e:
            st.error(f"Error creating debt charts: {e}")
            import traceback
            st.error(traceback.format_exc())
    else:
        st.info("Skipping charts as debt cashflow table is empty.")

    # --- Add download button for the table ---
    if not debt_df.empty:
        # Format for CSV export (optional: remove formatting for pure numbers)
        csv_df = debt_df.copy()
        # Example: Convert formatted numbers back to float for CSV if needed
        # for col in csv_df.columns[1:]: # Skip 'Category'
        #    csv_df[col] = csv_df[col].apply(lambda x: pd.to_numeric(str(x).replace(',', '').replace('(', '-').replace(')', '').replace('%',''), errors='ignore'))

        csv_data = csv_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download Debt Cashflow as CSV",
            data=csv_data,
            file_name="debt_cashflow.csv",
            mime="text/csv",
        )
    else:
        st.info("No data to download.")


def calculate_balances(cashflow_df, target_columns):
    """
    Returns monthly opening and closing balances based on cash movements.
    Assumes positive = disbursement / rolled interest, negative = repayment.
    Also handles forecasting for future periods using compound interest formula.
    """
    from config import DEFAULT_CONFIG

    opening_row = {"Category": "Opening balance"}
    closing_row = {"Category": "Closing balance, incl rolled interest"}

    monthly_cols = [col for col in target_columns if col not in OVERVIEW_COLUMNS]
    prev_closing = 0.0
    start_date = DEFAULT_CONFIG.get('start_date')

    # Get the base rate for forecasting periods
    annual_base_rates = DEFAULT_CONFIG.get('annual_base_rates', {})

    for col in monthly_cols:
        try:
            period_dt = datetime.strptime(col, SAV_DATE_FORMAT)
            is_forecast = period_dt >= start_date
        except ValueError:
            is_forecast = False

        opening_row[col] = prev_closing

        if is_forecast:
            # For forecast periods, use compound interest formula
            # Get base rate for this period
            base_rate = annual_base_rates.get(
                datetime(period_dt.year, period_dt.month, 1),
                DEFAULT_CONFIG.get('annual_base_rate', 0.045)
            )

            # Apply formula: closing = opening * (1 + rate)^(1/12)
            # Plus any additional cash movements for the period
            movement_sum = cashflow_df[col].sum()

            # Apply compound interest to the opening balance
            interest_factor = (1 + base_rate) ** (1 / 12) - 1
            interest_amount = prev_closing * interest_factor

            # For forecasting, we assume interest is capitalized (added to balance)
            closing_row[col] = prev_closing + interest_amount + movement_sum
        else:
            # For historical periods, just sum the movements
            movement_sum = cashflow_df[col].sum()
            closing_row[col] = prev_closing + movement_sum

        prev_closing = closing_row[col]

    return opening_row, closing_row


def forecast_loan_movements(loan_data, opening_balance, facility_amount, all_periods, start_date, categories):
    """
    Forecasts loan movements for future periods using the compound interest formula:
    -1 * (total facility + opening balance) * ((1 + annual_rate)**(1/12) - 1)

    Args:
        loan_data: DataFrame containing historical loan data
        opening_balance: Starting balance for forecasting
        facility_amount: Total loan facility amount
        all_periods: DatetimeIndex with all periods to forecast
        start_date: Start date for forecasting
        categories: Dictionary of financing categories with their rates

    Returns:
        Updated loan_data DataFrame with forecasted values
    """
    from config import DEFAULT_CONFIG

    # Get the annual base rates from config
    annual_base_rates = DEFAULT_CONFIG.get('annual_base_rates', {})

    # Create a copy of the loan data to avoid modifying the original
    forecast_loan_data = loan_data.copy()

    # Current balance starts at opening balance
    current_balance = opening_balance

    # Process each forecast period
    for period in all_periods:
        if period < start_date:
            continue

        period_str = period.strftime(SAV_DATE_FORMAT)
        period_date = datetime(period.year, period.month, 1)

        # Get the base rate for this period
        base_rate = annual_base_rates.get(period_date, DEFAULT_CONFIG.get('annual_base_rate', 0.045))

        # Calculate for each financing category
        for category, rate_info in categories.items():
            if category not in forecast_loan_data.index:
                continue

            # Get the effective rate (base rate + margin if applicable)
            effective_rate = rate_info['rate']
            if rate_info.get('apply_base_rate', False):
                effective_rate += base_rate

            # Calculate using the compound interest formula
            monthly_factor = (1 + effective_rate) ** (1 / 12) - 1

            # Apply to the appropriate amount (balance or undrawn)
            if rate_info.get('apply_to_undrawn', False):
                # For non-utilization fees, calculate on undrawn amount
                undrawn_amount = max(0, facility_amount - current_balance)
                cost = undrawn_amount * monthly_factor
            else:
                # For interest and other fees, calculate on current balance
                cost = current_balance * monthly_factor

                # If this is capitalized interest, update the current balance
                if category == 'Capitalised Interest' and rate_info.get('capitalize', False):
                    current_balance += cost

            # Apply the cost (negative value as it's an outflow)
            forecast_loan_data.loc[category, period_str] = -cost

    return forecast_loan_data


def create_debt_cashflow_table(senior_loan_df, mezzanine_loan_df):
    """Create a debt cashflow breakdown table with time-series columns."""
    from config import DEFAULT_CONFIG
    import pandas as pd

    # --- Configuration & Periods ---
    refinancing_date = DEFAULT_CONFIG.get('refinancing_date')
    all_periods = get_all_periods(
        ACQUISITION_DATE, START_DATE, END_DATE,
        DEFAULT_CONFIG.get('forecast_periods_count', 4)
    )
    period_cols = [p.strftime(SAV_DATE_FORMAT) for p in all_periods]
    target_columns = OVERVIEW_COLUMNS + period_cols
    annual_rates = DEFAULT_CONFIG.get('annual_base_rates', {})

    # Define financing categories with their rates
    senior_financing_categories = {
        'Capitalised Interest': {
            'rate': 0.03,  # Senior margin
            'apply_base_rate': True,
            'capitalize': True
        },
        'Non-utilisation Fee': {
            'rate': 0.015,  # 1.5% on undrawn
            'apply_base_rate': False,
            'apply_to_undrawn': True
        },
        'Fees': {
            'rate': 0.0025,  # 0.25% annual
            'apply_base_rate': False
        },
        'IMS Fees': {
            'rate': 0.0005,  # 0.05% annual
            'apply_base_rate': False
        }
    }

    mezzanine_financing_categories = {
        'Capitalised Interest': {
            'rate': 0.08,  # Mezzanine margin
            'apply_base_rate': True,
            'capitalize': True
        },
        'Non-utilisation Fee': {
            'rate': 0.02,  # 2% on undrawn
            'apply_base_rate': False,
            'apply_to_undrawn': True
        },
        'Fees': {
            'rate': 0.003,  # 0.3% annual
            'apply_base_rate': False
        },
        'IMS Fees': {
            'rate': 0.0005,  # 0.05% annual
            'apply_base_rate': False
        }
    }

    def update_or_append_row(df, new_row, category_col="Category"):
        df = df.copy()
        new_row_df = pd.DataFrame([new_row]).set_index(category_col)

        for col in new_row_df.columns:
            if col not in df.columns:
                df[col] = 0.0
        for col in df.columns:
            if col not in new_row_df.columns:
                new_row_df[col] = 0.0

        new_row_df = new_row_df[df.columns]

        if new_row_df.index[0] in df.index:
            df.loc[new_row_df.index[0]] = new_row_df.iloc[0]
        else:
            df = pd.concat([df, new_row_df])

        return df

    rows = []

    # --- Annual Base Rate Row ---
    rate_row = {"Category": "Annual Base Rate"}
    for col in target_columns:
        if col in OVERVIEW_COLUMNS:
            rate_row[col] = ""
        else:
            try:
                period_dt = datetime.strptime(col, SAV_DATE_FORMAT)
                rate = annual_rates.get(datetime(period_dt.year, period_dt.month, 1), "")
                rate_row[col] = f"{rate:.2%}" if isinstance(rate, float) else ""
            except ValueError:
                rate_row[col] = ""
    rows.append(rate_row)

    # === OakNorth Loan Section ===
    rows.append({"Category": "OakNorth Loan", **{col: "" for col in target_columns}})

    senior_data = preprocess_loan_df(
        senior_loan_df, 'senior_loan', all_periods,
        OVERVIEW_COLUMNS, START_DATE, CUTOFF_DATE, value_col='CASH OUT'
    )

    senior_cash_row = calculate_cash_payments(senior_loan_df, all_periods)
    senior_cash_row.update({
        col: sum(senior_cash_row.get(c, 0.0) for c in period_cols)
        for col in OVERVIEW_COLUMNS
    })

    senior_data = update_or_append_row(senior_data, senior_cash_row)

    # Get initial opening balance
    oak_opening, _ = calculate_balances(senior_data, target_columns)

    # Apply forecasting to senior loan
    # Get the last historical balance to use as starting point for forecasting
    senior_facility = DEFAULT_CONFIG.get('input_assumptions', {}).get('Senior Facility', 40_000_000)
    last_historical_period = max([
        datetime.strptime(col, SAV_DATE_FORMAT) for col in period_cols
        if datetime.strptime(col, SAV_DATE_FORMAT) < START_DATE
    ], default=ACQUISITION_DATE)
    last_historical_period_str = last_historical_period.strftime(SAV_DATE_FORMAT)

    # Get the opening balance for the first forecast period
    opening_balance = oak_opening.get(last_historical_period_str, 0.0)
    if isinstance(opening_balance, str) and opening_balance.strip() == "":
        opening_balance = 0.0

    # Apply forecasting to senior loan data
    senior_data = forecast_loan_movements(
        senior_data, opening_balance, senior_facility,
        all_periods, START_DATE, senior_financing_categories
    )

    updated_senior_data = senior_data.copy()

    def calculate_redemption_row(opening_row):
        categories_to_sum = [
            'Acquisition', 'Fees', 'Development', 'Capitalised Interest',
            'Non-utilisation Fee', 'IMS Fees', 'Cash Payment'
        ]
        redemption_row = {"Category": "Loan Redemption"}
        for col in target_columns:
            if col in OVERVIEW_COLUMNS:
                continue
            try:
                period_dt = datetime.strptime(col, SAV_DATE_FORMAT)
            except ValueError:
                redemption_row[col] = ""
                continue
            if period_dt.date() == refinancing_date.date():
                subtotal = sum([
                    updated_senior_data.loc[cat, col]
                    for cat in categories_to_sum
                    if cat in updated_senior_data.index
                ])
                opening_val = opening_row.get(col, 0.0)
                if isinstance(opening_val, str) and opening_val.strip() == "":
                    opening_val = 0.0
                redemption_row[col] = -(opening_val + subtotal)
            else:
                redemption_row[col] = 0.0

        redemption_row.update({
            col: sum(redemption_row.get(c, 0.0) for c in period_cols if c in redemption_row and not (
                        isinstance(redemption_row[c], str) and redemption_row[c].strip() == ""))
            for col in OVERVIEW_COLUMNS
        })
        return redemption_row

    loan_redemption_row = calculate_redemption_row(oak_opening)
    updated_senior_data = update_or_append_row(updated_senior_data, loan_redemption_row)

    oak_subtotal_row = {
        "Category": "Sub Total",
        **{col: updated_senior_data[col].sum() for col in target_columns}
    }

    oak_opening, oak_closing = calculate_balances(updated_senior_data, target_columns)
    updated_senior_data = pd.concat([
        updated_senior_data,
        pd.DataFrame([oak_subtotal_row]).set_index('Category')
    ])
    for category, row in updated_senior_data.iterrows():
        rows.append({"Category": category, **row.to_dict()})

    rows.extend([oak_opening, oak_closing])

    # === Coutts Loan Section ===
    rows.append({"Category": "Coutts Loan", **{col: "" for col in target_columns}})

    mezzanine_data = preprocess_loan_df(
        mezzanine_loan_df, 'mezzanine_loan', all_periods,
        OVERVIEW_COLUMNS, START_DATE, CUTOFF_DATE, value_col='CASH OUT'
    )

    mezzanine_cash_row = calculate_cash_payments(mezzanine_loan_df, all_periods)
    mezzanine_cash_row.update({
        col: sum(mezzanine_cash_row.get(c, 0.0) for c in period_cols)
        for col in OVERVIEW_COLUMNS
    })
    mezzanine_data = update_or_append_row(mezzanine_data, mezzanine_cash_row)

    # Get initial opening balance for mezzanine loan
    coutts_opening, _ = calculate_balances(mezzanine_data, target_columns)

    # Apply forecasting to mezzanine loan
    mezzanine_facility = DEFAULT_CONFIG.get('input_assumptions', {}).get('Mezzanine Facility', 15_000_000)
    # Get the opening balance for the first forecast period
    mezz_opening_balance = coutts_opening.get(last_historical_period_str, 0.0)
    if isinstance(mezz_opening_balance, str) and mezz_opening_balance.strip() == "":
        mezz_opening_balance = 0.0

    # Apply forecasting to mezzanine loan data
    mezzanine_data = forecast_loan_movements(
        mezzanine_data, mezz_opening_balance, mezzanine_facility,
        all_periods, START_DATE, mezzanine_financing_categories
    )

    # Recalculate balances after forecasting
    coutts_opening, coutts_closing = calculate_balances(mezzanine_data, target_columns)
    mezzanine_subtotal_row = {
        "Category": "Sub Total",
        **{col: mezzanine_data[col].sum() for col in target_columns}
    }
    mezzanine_data = pd.concat([
        mezzanine_data,
        pd.DataFrame([mezzanine_subtotal_row]).set_index('Category')
    ])
    for category, row in mezzanine_data.iterrows():
        rows.append({"Category": category, **row.to_dict()})

    rows.extend([coutts_opening, coutts_closing])

    final_columns_order = ['Category'] + target_columns
    return pd.DataFrame(rows)[final_columns_order]
# def generate_debt_cashflow_df(senior_loan_df, mezzanine_loan_df, all_periods):
#     columns = OVERVIEW_COLUMNS + [p.strftime(SAV_DATE_FORMAT) for p in all_periods]
#     categories = CATEGORIES["senior_loan"] + CATEGORIES["mezzanine_loan"] + ["Opening balance", "Closing balance", "Cash Payments"]
#     df = pd.DataFrame(index=categories, columns=columns).fillna(0.0)
#
#     def clean(val):
#         if pd.isna(val): return 0.0
#         if isinstance(val, str):
#             val = val.replace(',', '').replace('£', '').strip()
#             if val.startswith("(") and val.endswith(")"):
#                 val = "-" + val[1:-1]
#         try:
#             return float(val)
#         except: return 0.0
#
#     def populate_from_loan(source_df, loan_type_prefix):
#         for period in all_periods:
#             period_str = period.strftime(SAV_DATE_FORMAT)
#             month_data = source_df[source_df["MONTH"] == period].copy()
#             month_data["CASH OUT"] = month_data["CASH OUT"].apply(clean)
#             month_data["CASH IN"] = month_data["CASH IN"].apply(clean)
#
#             for category in CATEGORIES[loan_type_prefix]:
#                 is_cash_out = category not in ["Drawdown"]
#                 value = 0.0
#                 if is_cash_out:
#                     value = -month_data.loc[month_data["DETAIL 2"] == category, "CASH OUT"].sum()
#                 else:
#                     value = month_data.loc[month_data["DETAIL 2"] == category, "CASH IN"].sum()
#                 df.loc[category, period_str] += value
#
#             df.loc["Cash Payments", period_str] += month_data["CASH IN"].sum()
#
#     populate_from_loan(senior_loan_df, "senior_loan")
#     populate_from_loan(mezzanine_loan_df, "mezzanine_loan")
#
#     # Totals across time
#     inception_cutoff = DEFAULT_CONFIG['cutoff_date']
#     start_date = DEFAULT_CONFIG['start_date']
#
#     for category in df.index:
#         if category == "Actual/Forecast": continue
#         inception_cols = [p.strftime(SAV_DATE_FORMAT) for p in all_periods if p <= inception_cutoff]
#         forecast_cols = [p.strftime(SAV_DATE_FORMAT) for p in all_periods if p >= start_date]
#         df.loc[category, OVERVIEW_COLUMNS[0]] = df.loc[category, inception_cols].sum()
#         df.loc[category, OVERVIEW_COLUMNS[1]] = df.loc[category, forecast_cols].sum()
#         df.loc[category, "Total"] = df.loc[category, OVERVIEW_COLUMNS[0]] + df.loc[category, OVERVIEW_COLUMNS[1]]
#
#     return df
def forecast_loan_movements(loan_data, opening_balance, facility_amount, all_periods,
                            start_date, categories, loan_type='mezzanine', refinancing_date=None):
    """
    Forecasts loan movements for future periods using the compound interest formula:
    -1 * (total facility + opening balance) * ((1 + annual_rate)**(1/12) - 1)

    Args:
        loan_data: DataFrame containing historical loan data
        opening_balance: Starting balance for forecasting
        facility_amount: Total loan facility amount
        all_periods: DatetimeIndex with all periods to forecast
        start_date: Start date for forecasting
        categories: Dictionary of financing categories with their rates
        loan_type: 'senior' or 'mezzanine' - affects refinancing behavior
        refinancing_date: Date when refinancing occurs (senior loan ends)

    Returns:
        Updated loan_data DataFrame with forecasted values
    """
    from config import DEFAULT_CONFIG

    # Get the annual base rates from config
    annual_base_rates = DEFAULT_CONFIG.get('annual_base_rates', {})

    # Create a copy of the loan data to avoid modifying the original
    forecast_loan_data = loan_data.copy()

    # Current balance starts at opening balance
    current_balance = opening_balance

    # Process each forecast period
    for period in all_periods:
        if period < start_date:
            continue

        period_str = period.strftime(SAV_DATE_FORMAT)
        period_date = datetime(period.year, period.month, 1)

        # Check if we've reached refinancing date for senior loan
        if loan_type == 'senior' and refinancing_date and period_date >= refinancing_date:
            # For senior loan, after refinancing date, all values should be zero
            for category in categories.keys():
                if category in forecast_loan_data.index:
                    forecast_loan_data.loc[category, period_str] = 0.0

            # Balance is now zero
            current_balance = 0.0
            continue

        # Get the base rate for this period
        base_rate = annual_base_rates.get(period_date, DEFAULT_CONFIG.get('annual_base_rate', 0.045))

        # Calculate for each financing category
        for category, rate_info in categories.items():
            if category not in forecast_loan_data.index:
                continue

            # Skip if balance is zero (unless this is a 'Cash Payment' that increases the balance)
            if current_balance == 0 and category != 'Cash Payment':
                forecast_loan_data.loc[category, period_str] = 0.0
                continue

            # Get the effective rate (base rate + margin if applicable)
            effective_rate = rate_info['rate']
            if rate_info.get('apply_base_rate', False):
                effective_rate += base_rate

            # Calculate using the compound interest formula
            monthly_factor = (1 + effective_rate) ** (1 / 12) - 1

            # Apply to the appropriate amount (balance or undrawn)
            if rate_info.get('apply_to_undrawn', False):
                # For non-utilization fees, calculate on undrawn amount
                undrawn_amount = max(0, facility_amount - current_balance)
                cost = undrawn_amount * monthly_factor
            else:
                # For interest and other fees, calculate on current balance
                cost = current_balance * monthly_factor

            # Apply the cost (negative value as it's an outflow)
            forecast_loan_data.loc[category, period_str] = -cost

            # If this is capitalized interest, update the current balance
            if category == 'Capitalised Interest' and rate_info.get('capitalize', False):
                current_balance += cost

    # Update the overview columns after all period calculations
    for category in forecast_loan_data.index:
        forecast_cols = [p.strftime(SAV_DATE_FORMAT) for p in all_periods if p >= start_date]
        if len(forecast_cols) > 0 and OVERVIEW_COLUMNS[1] in forecast_loan_data.columns:
            forecast_loan_data.loc[category, OVERVIEW_COLUMNS[1]] = forecast_loan_data.loc[
                category, forecast_cols].sum()

        if OVERVIEW_COLUMNS[2] in forecast_loan_data.columns:
            forecast_loan_data.loc[category, OVERVIEW_COLUMNS[2]] = (
                    forecast_loan_data.loc[category, OVERVIEW_COLUMNS[0]]
                    + forecast_loan_data.loc[category, OVERVIEW_COLUMNS[1]]
            )

    return forecast_loan_data


def create_debt_cashflow_table(senior_loan_df, mezzanine_loan_df):
    """Create a debt cashflow breakdown table with time-series columns."""
    from config import DEFAULT_CONFIG
    import pandas as pd

    # --- Configuration & Periods ---
    refinancing_date = DEFAULT_CONFIG.get('refinancing_date')
    all_periods = get_all_periods(
        ACQUISITION_DATE, START_DATE, END_DATE,
        DEFAULT_CONFIG.get('forecast_periods_count', 4)
    )
    period_cols = [p.strftime(SAV_DATE_FORMAT) for p in all_periods]
    target_columns = OVERVIEW_COLUMNS + period_cols
    annual_rates = DEFAULT_CONFIG.get('annual_base_rates', {})

    # Define financing categories with their rates
    senior_financing_categories = {
        'Capitalised Interest': {
            'rate': DEFAULT_CONFIG.get('input_assumptions', {}).get('Senior Margin', 0.03),
            'apply_base_rate': True,
            'capitalize': True
        },
        'Non-utilisation Fee': {
            'rate': DEFAULT_CONFIG.get('input_assumptions', {}).get('Senior Non-utilisation Rate', 0.015),
            'apply_base_rate': False,
            'apply_to_undrawn': True
        },
        'Fees': {
            'rate': DEFAULT_CONFIG.get('input_assumptions', {}).get('Senior Fees Rate', 0.0025),
            'apply_base_rate': False
        },
        'IMS Fees': {
            'rate': DEFAULT_CONFIG.get('input_assumptions', {}).get('Senior IMS Rate', 0.0005),
            'apply_base_rate': False
        }
    }

    mezzanine_financing_categories = {
        'Capitalised Interest': {
            'rate': DEFAULT_CONFIG.get('input_assumptions', {}).get('Mezzanine Margin', 0.08),
            'apply_base_rate': True,
            'capitalize': True
        },
        'Non-utilisation Fee': {
            'rate': DEFAULT_CONFIG.get('input_assumptions', {}).get('Mezzanine Non-utilisation Rate', 0.02),
            'apply_base_rate': False,
            'apply_to_undrawn': True
        },
        'Fees': {
            'rate': DEFAULT_CONFIG.get('input_assumptions', {}).get('Mezzanine Fees Rate', 0.003),
            'apply_base_rate': False
        },
        'IMS Fees': {
            'rate': DEFAULT_CONFIG.get('input_assumptions', {}).get('Mezzanine IMS Rate', 0.0005),
            'apply_base_rate': False
        }
    }

    def update_or_append_row(df, new_row, category_col="Category"):
        df = df.copy()
        new_row_df = pd.DataFrame([new_row]).set_index(category_col)

        for col in new_row_df.columns:
            if col not in df.columns:
                df[col] = 0.0
        for col in df.columns:
            if col not in new_row_df.columns:
                new_row_df[col] = 0.0

        new_row_df = new_row_df[df.columns]

        if new_row_df.index[0] in df.index:
            df.loc[new_row_df.index[0]] = new_row_df.iloc[0]
        else:
            df = pd.concat([df, new_row_df])

        return df

    rows = []

    # --- Annual Base Rate Row ---
    rate_row = {"Category": "Annual Base Rate"}
    for col in target_columns:
        if col in OVERVIEW_COLUMNS:
            rate_row[col] = ""
        else:
            try:
                period_dt = datetime.strptime(col, SAV_DATE_FORMAT)
                rate = annual_rates.get(datetime(period_dt.year, period_dt.month, 1), "")
                rate_row[col] = f"{rate:.2%}" if isinstance(rate, float) else ""
            except ValueError:
                rate_row[col] = ""
    rows.append(rate_row)

    # === OakNorth Loan Section ===
    rows.append({"Category": "OakNorth Loan", **{col: "" for col in target_columns}})

    senior_data = preprocess_loan_df(
        senior_loan_df, 'senior_loan', all_periods,
        OVERVIEW_COLUMNS, START_DATE, CUTOFF_DATE, value_col='CASH OUT'
    )
    senior_cash_row = calculate_cash_payments(senior_loan_df, all_periods)
    senior_cash_row.update({
        col: sum(senior_cash_row.get(c, 0.0) for c in period_cols)
        for col in OVERVIEW_COLUMNS
    })

    senior_data = update_or_append_row(senior_data, senior_cash_row)

    # Get initial opening balance
    oak_opening, _ = calculate_balances(senior_data, target_columns)

    # Apply forecasting to senior loan
    # Get the last historical balance to use as starting point for forecasting
    senior_facility = DEFAULT_CONFIG.get('input_assumptions', {}).get('Senior Facility', 40_000_000)
    last_historical_period = max([
        datetime.strptime(col, SAV_DATE_FORMAT) for col in period_cols
        if datetime.strptime(col, SAV_DATE_FORMAT) < START_DATE
    ], default=ACQUISITION_DATE)
    last_historical_period_str = last_historical_period.strftime(SAV_DATE_FORMAT)

    # Get the opening balance for the first forecast period
    opening_balance = oak_opening.get(last_historical_period_str, 0.0)
    if isinstance(opening_balance, str) and opening_balance.strip() == "":
        opening_balance = 0.0

    # Apply forecasting to senior loan data - note we specify loan_type as 'senior'
    # and pass the refinancing_date
    senior_data = forecast_loan_movements(
        senior_data, opening_balance, senior_facility,
        all_periods, START_DATE, senior_financing_categories,
        loan_type='senior', refinancing_date=refinancing_date
    )

    updated_senior_data = senior_data.copy()

    def calculate_redemption_row(opening_row):
        categories_to_sum = [
            'Acquisition', 'Fees', 'Development', 'Capitalised Interest',
            'Non-utilisation Fee', 'IMS Fees', 'Cash Payment'
        ]
        redemption_row = {"Category": "Loan Redemption"}
        for col in target_columns:
            if col in OVERVIEW_COLUMNS:
                continue
            try:
                period_dt = datetime.strptime(col, SAV_DATE_FORMAT)
            except ValueError:
                redemption_row[col] = ""
                continue
            if period_dt.date() == refinancing_date.date():
                subtotal = sum([
                    updated_senior_data.loc[cat, col]
                    for cat in categories_to_sum
                    if cat in updated_senior_data.index
                ])
                opening_val = opening_row.get(col, 0.0)
                if isinstance(opening_val, str) and opening_val.strip() == "":
                    opening_val = 0.0
                redemption_row[col] = -(opening_val + subtotal)
            else:
                redemption_row[col] = 0.0

        redemption_row.update({
            col: sum(redemption_row.get(c, 0.0) for c in period_cols if c in redemption_row and not (
                        isinstance(redemption_row[c], str) and redemption_row[c].strip() == ""))
            for col in OVERVIEW_COLUMNS
        })
        return redemption_row

    loan_redemption_row = calculate_redemption_row(oak_opening)
    updated_senior_data = update_or_append_row(updated_senior_data, loan_redemption_row)

    oak_subtotal_row = {
        "Category": "Sub Total",
        **{col: updated_senior_data[col].sum() for col in target_columns}
    }
    # Ensure forecast total is zero if refinancing is before forecast period
    if refinancing_date and refinancing_date < START_DATE:
        oak_subtotal_row[OVERVIEW_COLUMNS[1]] = 0.0
        oak_subtotal_row[OVERVIEW_COLUMNS[2]] = oak_subtotal_row[OVERVIEW_COLUMNS[0]]

    oak_opening, oak_closing = calculate_balances(updated_senior_data, target_columns)
    updated_senior_data = pd.concat([
        updated_senior_data,
        pd.DataFrame([oak_subtotal_row]).set_index('Category')
    ])
    for category, row in updated_senior_data.iterrows():
        rows.append({"Category": category, **row.to_dict()})

    rows.extend([oak_opening, oak_closing])

    # === Coutts Loan Section ===
    rows.append({"Category": "Coutts Loan", **{col: "" for col in target_columns}})

    mezzanine_data = preprocess_loan_df(
        mezzanine_loan_df, 'mezzanine_loan', all_periods,
        OVERVIEW_COLUMNS, START_DATE, CUTOFF_DATE, value_col='CASH OUT'
    )

    mezzanine_cash_row = calculate_cash_payments(mezzanine_loan_df, all_periods)
    mezzanine_cash_row.update({
        col: sum(mezzanine_cash_row.get(c, 0.0) for c in period_cols)
        for col in OVERVIEW_COLUMNS
    })
    mezzanine_data = update_or_append_row(mezzanine_data, mezzanine_cash_row)

    # Get initial opening balance for mezzanine loan
    coutts_opening, _ = calculate_balances(mezzanine_data, target_columns)

    # Apply forecasting to mezzanine loan
    mezzanine_facility = DEFAULT_CONFIG.get('input_assumptions', {}).get('Mezzanine Facility', 15_000_000)
    # Get the opening balance for the first forecast period
    mezz_opening_balance = coutts_opening.get(last_historical_period_str, 0.0)
    if isinstance(mezz_opening_balance, str) and mezz_opening_balance.strip() == "":
        mezz_opening_balance = 0.0

    # Apply forecasting to mezzanine loan data - specify loan_type as 'mezzanine'
    # (mezzanine loan continues after refinancing)
    mezzanine_data = forecast_loan_movements(
        mezzanine_data, mezz_opening_balance, mezzanine_facility,
        all_periods, START_DATE, mezzanine_financing_categories,
        loan_type='mezzanine', refinancing_date=refinancing_date
    )

    # Recalculate balances after forecasting
    coutts_opening, coutts_closing = calculate_balances(mezzanine_data, target_columns)
    mezzanine_subtotal_row = {
        "Category": "Sub Total",
        **{col: mezzanine_data[col].sum() for col in target_columns}
    }
    mezzanine_data = pd.concat([
        mezzanine_data,
        pd.DataFrame([mezzanine_subtotal_row]).set_index('Category')
    ])
    for category, row in mezzanine_data.iterrows():
        rows.append({"Category": category, **row.to_dict()})

    rows.extend([coutts_opening, coutts_closing])

    final_columns_order = ['Category'] + target_columns
    return pd.DataFrame(rows)[final_columns_order]