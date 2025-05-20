import pandas as pd
import plotly.express as px
import streamlit as st
from dateutil.relativedelta import relativedelta

from config import CATEGORIES
from config import DEFAULT_CONFIG
from monte_carlo import run_monte_carlo_simulation, create_simulation_plots
from utils import format_cf_value
from debt_cashflow import display_debt_cashflow, create_debt_charts
from cashflow.return_summary import ReturnSummaryCalculator

SAV_DATE_FORMAT = '%b-%y'

OVERVIEW_COLUMNS = [f"Inception to {DEFAULT_CONFIG['cutoff_date'].strftime(SAV_DATE_FORMAT)}",
                    f"{DEFAULT_CONFIG['start_date'].strftime(SAV_DATE_FORMAT)} to Exit", 'Total']
# Import display functions from debt_cashflow.py

FORECAST_START_DATE = DEFAULT_CONFIG['cutoff_date'] + relativedelta(months=1)

ACQUISITION_DATE = DEFAULT_CONFIG['acquisition_date']
CUTOFF_DATE = DEFAULT_CONFIG.get('cutoff_date') + relativedelta(day=31)
DEVELOPMENT_COSTS_STR = 'Development costs'
ANTVIC_STR = 'antvic'
QUARTERLY_MONTH_INDEX = 2  # Third month in quarter (0-based index)


def add_return_summary_tab(summary_calculator):
    """Add the return summary tab using the calculator."""
    st.subheader("Return Summary")

    # Create three columns for the layout
    col1, col2, col3 = st.columns(3)

    # Exit Value table
    with col1:
        st.write("### Exit Value")
        st.table(summary_calculator.get_exit_value_df())

    # Costs table
    with col2:
        st.write("### Costs")
        st.table(summary_calculator.get_costs_df())

    # Third column left empty for now
    with col3:
        pass

def add_profit_split_tab(summary_calculator, equity_split_data):
        """Add the profit split tab using the calculator."""
        st.subheader("Profit Split")

        # Example: Display profit calculation
        profit = summary_calculator.get_profit()
        st.write(f"### Total Profit: £{profit:,.0f}")

        # Here you'll add the profit split logic using both the summary_calculator
        # and equity_split_data
        # ...
def display_equity_split(equity_split_data):
    """Display a compact visualization of equity split percentages."""
    if not equity_split_data:
        return

    st.subheader("Equity Split (% of Total)")

    # Create a two-column layout
    col1, col2, col3 = st.columns([1, 1, 1])  # 1:2 ratio gives more space to the chart

    # Create a vertical table in the first column
    with col1:
        # Convert to DataFrame for a clean tabular display
        table_data = {
            "Shareholder": list(equity_split_data.keys()),
            "Percentage": [f"{percentage:.1f}%" for percentage in equity_split_data.values()]
        }
        split_df = pd.DataFrame(table_data)
        st.table(split_df)  # Using st.table for a static display without sorting

    # Add a pie chart visualization in the second column
    with col2:
        fig = px.pie(
            values=list(equity_split_data.values()),
            names=list(equity_split_data.keys()),
            # title="Equity Split",
            hole=0.4,  # Makes it as donut chart
            width=500, height=500
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            # legend=dict(orientation="h", yanchor="bottom", y=0),
            margin=dict(t=0, b=0, l=0, r=0)  # Tighter margins
        )
        st.plotly_chart(fig, use_container_width=False)

def create_dashboard(cashflow_data: pd.DataFrame, cashflow_generator=None,
                     annual_base_rate: float = 0.001, debt_cashflow_df=None, equity_cashflow_df=None,
                     mc_config: dict = None, equity_split_data=None) -> None:
    """Create the Streamlit dashboard with cashflow data and Monte Carlo simulations."""
    st.title('Cashflow Analysis Dashboard')
    summary_calculator = ReturnSummaryCalculator(cashflow_data, DEFAULT_CONFIG)

    # Create tabs for different dashboard sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Project Cashflow",
        "Debt Cashflow",
        "Equity Cashflow",
        "Return Summary",
        "Monte Carlo Simulation",
        "Profit Analysis"
    ])

    with tab1:
        create_cashflow_analysis(cashflow_data)

    with tab2:
        # Use the debt cashflow from the generator instead of creating it here
        display_debt_cashflow(debt_cashflow_df)

        # Create debt charts if data is available
        if not cashflow_generator.debt_cashflow_df.empty:
            create_debt_charts(cashflow_generator.debt_cashflow_df)

    with tab3:
        display_debt_cashflow(equity_cashflow_df)
        if equity_split_data:
            display_equity_split(equity_split_data)
    with tab4:
        add_return_summary_tab(summary_calculator)
    with tab5:
        if mc_config:
            create_monte_carlo_analysis(cashflow_data, mc_config)
        else:
            st.warning("Monte Carlo simulation configuration not provided.")

    with tab6:
        if mc_config and 'profit_details' in mc_config:
            create_profit_analysis(cashflow_data, mc_config['profit_details'])
        else:
            st.warning("Profit details not provided in the configuration.")


def create_cashflow_analysis(cashflow_data: pd.DataFrame, equity_split_data=None) -> None:
    """Create the project cashflow section of the dashboard."""
    # Define the hierarchy based on the screenshot and CATEGORIES
    hierarchy = {
        'Acquisition Total': CATEGORIES['acquisition_components'],
        'Development': CATEGORIES['development_components'],
        'Financing costs': CATEGORIES['financing'],
        'Total Project Cashflow': CATEGORIES['total_project_components'],
    }

    # Create a new DataFrame to display with hierarchical structure
    display_df = pd.DataFrame(columns=cashflow_data.columns)

    # Build the hierarchical DataFrame
    main_categories = set(hierarchy.keys())
    for main_category, subcategories in hierarchy.items():
        if main_category in cashflow_data.index:
            display_df.loc[main_category] = cashflow_data.loc[main_category]
        else:
            display_df.loc[main_category] = 0.0

        for subcat in subcategories:
            if subcat in cashflow_data.index:
                indented_subcat = f'    {subcat}'
                display_df.loc[indented_subcat] = cashflow_data.loc[subcat]

    for category in cashflow_data.index:
        if not any(category in subcats for subcats in hierarchy.values()) and \
                category not in hierarchy:
            display_df.loc[category] = cashflow_data.loc[category]

    # Detailed Cashflow Data with bold styling for main categories
    st.subheader('Project Cashflow')

    # Format the DataFrame for display
    styled_df = display_df.copy()
    for col in styled_df.columns:
        styled_df[col] = styled_df.apply(
            lambda row: row[col] if 'Actual/Forecast' in row.name else format_cf_value(row[col]), axis=1)

    # Define CSS styles to keep lines shorter
    header_style = 'border: 1px solid #ddd; padding: 8px; min-width: 120px;'
    sticky_style = 'position: sticky; z-index: 1;'
    column_header_style = f'{header_style} text-align: right;'
    row_style = 'border: 1px solid #ddd; padding: 8px; min-width: 120px;'

    # Define the columns to freeze (up to "Total")
    frozen_columns = OVERVIEW_COLUMNS
    # Calculate cumulative left positions for sticky columns
    # Assume each column has a fixed width (e.g., 120px)
    column_width = 119  # Adjust this value if needed based on actual column width
    left_positions = {col: (i + 1) * column_width for i, col in enumerate(frozen_columns)}
    # Category column will be at left: 0px, so we start from the next column

    # Generate HTML table with sticky columns up to "Total"
    html = '<div style="overflow-x: auto; max-width: 100%;">'
    html += '<table style="width:100%; border-collapse: collapse; margin: 0;">'

    # Add header row
    html += '<tr style="background-color: #f2f2f2;">'
    # Category column (first column, always sticky)
    category_header_style = f'{header_style} text-align: left; ' \
                            f'background-color: #f2f2f2; {sticky_style} ' \
                            f'left: 0px;'
    html += f'<th style="{category_header_style}">Category</th>'
    # Other columns
    for col in styled_df.columns:
        if col in frozen_columns:
            left_pos = left_positions[col]
            col_header_style = f'{header_style} text-align: right; ' \
                               f'background-color: #f2f2f2; {sticky_style} ' \
                               f'left: {left_pos}px;'
            html += f'<th style="{col_header_style}">{col}</th>'
        else:
            html += f'<th style="{column_header_style}">{col}</th>'
    html += '</tr>'

    for idx, row in styled_df.iterrows():
        is_main_category = idx in main_categories
        is_actual_forecast = idx == 'Actual/Forecast'
        font_weight = 'bold' if is_main_category else 'normal'
        bg_color = '#e6f3ff' if is_main_category else '#ffffff'

        html += f'<tr style="background-color: {bg_color};">'
        # Category column (first column, always sticky)
        category_style = f'{row_style} font-weight: {font_weight}; ' \
                         f'white-space: pre; {sticky_style} ' \
                         f'background-color: {bg_color}; left: 0px;'
        html += f'<td style="{category_style}">{idx}</td>'
        # Other columns
        for col, val in row.items():
            text_align = 'left' if is_actual_forecast else 'right'
            if col in frozen_columns:
                left_pos = left_positions[col]
                cell_style = f'{row_style} text-align: {text_align}; ' \
                             f'background-color: {bg_color}; {sticky_style} ' \
                             f'left: {left_pos}px;'
                html += f'<td style="{cell_style}">{val}</td>'
            else:
                cell_style = f'{row_style} text-align: {text_align};'
                html += f'<td style="{cell_style}">{val}</td>'
        html += '</tr>'

    html += '</table>'
    html += '</div>'

    st.markdown(html, unsafe_allow_html=True)
    st.subheader('Download Cashflow Table')
    csv = cashflow_data.to_csv(index=True)
    st.download_button(
        label="Download Cashflow as CSV",
        data=csv,
        file_name="cashflow_table.csv",
        mime="text/csv",
    )

    # Sidebar Filters for Cashflow Analysis
    st.sidebar.subheader('Cashflow Filters')
    selectable_categories = list(hierarchy) + [
        cat for cat in cashflow_data.index
        if cat not in sum(hierarchy.values(), []) and cat not in hierarchy
    ]
    selected_categories = st.sidebar.multiselect(
        'Select Categories',
        options=[cat for cat in selectable_categories if cat != 'Actual/Forecast'],
        default=['Total Project Cashflow', 'Development'],
        key="cashflow_categories"
    )

    plot_df = cashflow_data.loc[selected_categories].T
    period_columns = [col for col in plot_df.index if col not in frozen_columns]
    plot_df_periods = plot_df.loc[period_columns]
    plot_df_periods_numeric = plot_df_periods.apply(pd.to_numeric, errors='coerce')

    st.subheader('Cumulative Cashflow Over Time')
    fig1 = px.line(plot_df_periods_numeric.cumsum(),
                   title='Cumulative Cashflow',
                   labels={'value': 'Amount (£)', 'index': 'Period'})
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader('Monthly Cashflow')
    fig2 = px.bar(plot_df_periods_numeric,
                  title='Monthly Cashflow by Category',
                  labels={'value': 'Amount (£)', 'index': 'Period'},
                  barmode='group')
    st.plotly_chart(fig2, use_container_width=True)


def create_monte_carlo_analysis(cashflow_data: pd.DataFrame, mc_config: dict) -> None:
    """  Create the Monte Carlo simulation analysis section of the dashboard."""
    st.subheader('Monte Carlo Simulation Analysis')

    # Input parameters for the simulation
    with st.expander("Monte Carlo Simulation Parameters", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Exit Value Parameters**")
            exit_value_min = st.number_input(
                "Min Exit Value (£)",
                value=float(mc_config['exit_value']['min']),
                step=1000000.0,
                format="%.0f"
            )
            exit_value_mean = st.number_input(
                "Mean Exit Value (£)",
                value=float(mc_config['exit_value']['mean']),
                step=1000000.0,
                format="%.0f"
            )
            exit_value_max = st.number_input(
                "Max Exit Value (£)",
                value=float(mc_config['exit_value']['max']),
                step=1000000.0,
                format="%.0f"
            )

            # Additional simulation parameters could be added here if needed

        with col2:
            st.markdown("**Cost Variation Parameters**")
            acquisition_min = st.number_input(
                "Acquisition Cost Min Variation",
                value=float(mc_config['cost_variation']['acquisition']['min']),
                step=0.01,
                format="%.2f"
            )
            acquisition_max = st.number_input(
                "Acquisition Cost Max Variation",
                value=float(mc_config['cost_variation']['acquisition']['max']),
                step=0.01,
                format="%.2f"
            )

            development_min = st.number_input(
                "Development Cost Min Variation",
                value=float(mc_config['cost_variation']['development']['min']),
                step=0.01,
                format="%.2f"
            )
            development_max = st.number_input(
                "Development Cost Max Variation",
                value=float(mc_config['cost_variation']['development']['max']),
                step=0.01,
                format="%.2f"
            )

            # Additional cost variation parameters could be added here if needed

    # Number of simulations slider
    num_simulations = st.slider(
        "Number of Simulations",
        min_value=100,
        max_value=10000,
        value=mc_config['num_simulations'],
        step=100
    )

    # Project timeline parameters
    with st.expander("Project Timeline Distribution", expanded=False):
        st.info("These parameters control how development costs are distributed over time. " +
                "The percentages should add up to 1.0 (100%).")

        col1, col2 = st.columns(2)
        with col1:
            early_dev = st.slider(
                "Early Development Phase",
                min_value=0.1,
                max_value=0.5,
                value=float(mc_config['project_timeline']['early_development']),
                step=0.05,
                format="%.2f"
            )

            mid_dev = st.slider(
                "Mid Development Phase",
                min_value=0.1,
                max_value=0.5,
                value=float(mc_config['project_timeline']['mid_development']),
                step=0.05,
                format="%.2f"
            )

        with col2:
            late_dev = st.slider(
                "Late Development Phase",
                min_value=0.1,
                max_value=0.5,
                value=float(mc_config['project_timeline']['late_development']),
                step=0.05,
                format="%.2f"
            )

            # Calculate acquisition automatically to ensure sum is 1.0
            acquisition = 1.0 - (early_dev + mid_dev + late_dev)
            st.metric(
                "Acquisition Phase",
                f"{acquisition:.2f}",
                help="This value is calculated automatically to ensure all phases sum to 100%"
            )

    # Update the configuration with user inputs
    updated_mc_config = {
        'num_simulations': num_simulations,
        'exit_value': {
            'mean': exit_value_mean,
            'min': exit_value_min,
            'max': exit_value_max
        },
        'cost_variation': {
            'acquisition': {
                'min': acquisition_min,
                'max': acquisition_max
            },
            'development': {
                'min': development_min,
                'max': development_max
            }
        },
        'project_timeline': {
            'acquisition': acquisition,
            'early_development': early_dev,
            'mid_development': mid_dev,
            'late_development': late_dev
        }
    }

    # Run the Monte Carlo simulation
    if st.button("Run Monte Carlo Simulation"):
        with st.spinner("Running simulation..."):
            results_df, summary_stats = run_monte_carlo_simulation(
                cashflow_data,
                updated_mc_config['num_simulations'],
                updated_mc_config['exit_value'],
                updated_mc_config['cost_variation'],
                updated_mc_config['project_timeline']
            )

            # Display summary statistics
            st.subheader("Simulation Results Summary")

            # Create two columns for IRR and POC stats
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**IRR Statistics**")
                irr_stats = pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Min', 'Max', 'Std Dev',
                               '5th Percentile', '25th Percentile', '75th Percentile', '95th Percentile'],
                    'Value': [
                        f"{summary_stats['IRR']['mean']:.2%}",
                        f"{summary_stats['IRR']['median']:.2%}",
                        f"{summary_stats['IRR']['min']:.2%}",
                        f"{summary_stats['IRR']['max']:.2%}",
                        f"{summary_stats['IRR']['std']:.2%}",
                        f"{summary_stats['IRR']['percentiles'][5]:.2%}",
                        f"{summary_stats['IRR']['percentiles'][25]:.2%}",
                        f"{summary_stats['IRR']['percentiles'][75]:.2%}",
                        f"{summary_stats['IRR']['percentiles'][95]:.2%}"
                    ]
                })
                st.dataframe(irr_stats, use_container_width=True)

            with col2:
                st.markdown("**Profit on Cost Statistics**")
                poc_stats = pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Min', 'Max', 'Std Dev',
                               '5th Percentile', '25th Percentile', '75th Percentile', '95th Percentile'],
                    'Value': [
                        f"{summary_stats['POC']['mean']:.2%}",
                        f"{summary_stats['POC']['median']:.2%}",
                        f"{summary_stats['POC']['min']:.2%}",
                        f"{summary_stats['POC']['max']:.2%}",
                        f"{summary_stats['POC']['std']:.2%}",
                        f"{summary_stats['POC']['percentiles'][5]:.2%}",
                        f"{summary_stats['POC']['percentiles'][25]:.2%}",
                        f"{summary_stats['POC']['percentiles'][75]:.2%}",
                        f"{summary_stats['POC']['percentiles'][95]:.2%}"
                    ]
                })
                st.dataframe(poc_stats, use_container_width=True)

            # Create and display the visualization plots
            plots = create_simulation_plots(results_df, summary_stats)

            # Display all plots
            st.plotly_chart(plots['irr_histogram'], use_container_width=True)
            st.plotly_chart(plots['poc_histogram'], use_container_width=True)
            st.plotly_chart(plots['irr_poc_scatter'], use_container_width=True)
            st.plotly_chart(plots['cost_boxplot'], use_container_width=True)

            # Offer to download the simulation results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Simulation Results",
                data=csv,
                file_name="monte_carlo_results.csv",
                mime="text/csv",
            )
    else:
        st.info("Click 'Run Monte Carlo Simulation' to see the results.")


def create_profit_analysis(cashflow_data, profit_details: dict) -> None:
    """  Create the profit analysis section of the dashboard."""
    st.subheader('Profit Analysis')

    # Create inputs for profit analysis parameters
    with st.expander("Profit Analysis Parameters", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Exit Value**")
            sell_price = st.number_input(
                "Sell Price (£)",
                value=float(profit_details['sell_price']),
                step=100000.0,
                format="%.0f",
                key="sell_price"
            )

            selling_costs = st.number_input(
                "Selling Costs (£)",
                value=sell_price * (profit_details['selling_cost_percentage']),
                step=1000.0,
                format="%.0f",
                key="selling_costs"
            )
            st.markdown("**Costs**")
            acquisition_costs = cashflow_data.loc['Acquisition Total', 'Total']
            development_costs = st.number_input(
                "Development Costs (£)",
                value=float(cashflow_data.loc['Total Development Costs', 'Total']),
                step=float(100000),
                format="%.0f",
                key="development_costs"
            )

        with col2:
            st.markdown("**Additional Costs**")
            # loan_interest = st.number_input(
            #     "Loan Interest (£)",
            #     value=float(profit_details['loan_interest']),
            #     step=10000.0,
            #     format="%.0f",
            #     key="loan_interest"
            # )
            corp_tax = st.number_input(
                "Corporation Tax (£)",
                value=sell_price * float(profit_details['corp_tax_percentage']),
                step=10000.0,
                format="%.0f",
                key="corp_tax"
            )

    # Calculate profit metrics
    net_sale_price = sell_price + selling_costs
    total_development_cost = acquisition_costs + development_costs
    profit = net_sale_price - abs(total_development_cost)
    profit_after_tax = profit + corp_tax

    # Calculate profit on cost
    poc = profit / abs(total_development_cost)
    poc_after_tax = profit_after_tax / abs(total_development_cost)

    # Display profit summary
    st.subheader("Profit Summary")

    # Create a DataFrame for profit summary
    profit_summary = pd.DataFrame({
        'Metric': [
            'Sell Price',
            'Selling Costs',
            'Net Sale Price',
            'Acquisition Costs',
            'Development Costs',

            'Total Development Cost',
            'Gross Profit',
            'Profit on Cost (POC)',
            'Corporation Tax',
            'Profit After Tax',
            'POC After Tax'
        ],
        'Value': [
            f"£{sell_price:,.0f}",
            f"£{selling_costs:,.0f}",
            f"£{net_sale_price:,.0f}",
            f"£{acquisition_costs:,.0f}",
            f"£{development_costs:,.0f}",

            f"£{total_development_cost:,.0f}",
            f"£{profit:,.0f}",
            f"{poc:.2%}",
            f"£{corp_tax:,.0f}",
            f"£{profit_after_tax:,.0f}",
            f"{poc_after_tax:.2%}"
        ]
    })

    # Display the profit summary as a styled dataframe
    st.dataframe(
        profit_summary,
        column_config={
            "Metric": st.column_config.TextColumn("Metric"),
            "Value": st.column_config.TextColumn("Value")
        },
        use_container_width=True,
        hide_index=True
    )

    # Create a waterfall chart for profit breakdown
    waterfall_data = {
        'Category': [
            'Net Sale Price',
            'Acquisition Costs',
            'Development Costs',
            'Gross Profit',
            'Corporation Tax',
            'Profit After Tax'
        ],
        'Amount': [
            net_sale_price,
            acquisition_costs,
            development_costs,
            profit,
            corp_tax,
            profit_after_tax
        ],
        'Type': [
            'total',
            'relative',
            'relative',
            'total',
            'relative',
            'total'
        ]
    }

    # Convert to a DataFrame
    waterfall_df = pd.DataFrame(waterfall_data)

    # Create a custom color map
    colors = {
        'Net Sale Price': 'green',
        'Acquisition Costs': 'red',
        'Development Costs': 'red',
        'Gross Profit': 'blue',
        'Corporation Tax': 'red',
        'Profit After Tax': 'purple'
    }

    # Assign colors to each category
    waterfall_df['color'] = waterfall_df['Category'].map(colors)

    # Create the waterfall chart
    fig = px.bar(
        waterfall_df,
        x='Category',
        y='Amount',
        title='Profit Waterfall Chart',
        color='Category',
        color_discrete_map=colors
    )

    # Add a horizontal line at y=0
    fig.add_shape(
        type='line',
        x0=-0.5,
        x1=len(waterfall_df) - 0.5,
        y0=0,
        y1=0,
        line=dict(color='black', width=1, dash='dash')
    )

    # Format y-axis tick labels as currency
    fig.update_yaxes(tickprefix='£', tickformat=',')

    # Display the waterfall chart
    st.plotly_chart(fig, use_container_width=True)
