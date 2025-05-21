import pandas as pd
import plotly.express as px
import streamlit as st

from config import DEFAULT_CONFIG
from utils import format_cf_value

SAV_DATE_FORMAT = '%b-%y'
OVERVIEW_COLUMNS = [f"Inception to {DEFAULT_CONFIG['cutoff_date'].strftime(SAV_DATE_FORMAT)}",
                    f"{DEFAULT_CONFIG['start_date'].strftime(SAV_DATE_FORMAT)} to Exit", 'Total']


def display_debt_cashflow(debt_df, button_key=None):
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
    # Add a download button below
    csv = debt_df.to_csv().encode('utf-8')
    st.download_button(
        label=f"Download {button_key.capitalize()} Cashflow as CSV",
        data=csv,
        file_name="debt_cashflow.csv",
        mime="text/csv",
        key=button_key
    )


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
