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
            "Percentage": [f"{percentage:.1f}%" for percentage in equity_split_data.values()],
            "Financing rate": [f"{percentage * 100:.1f}%" for percentage in DEFAULT_CONFIG['financing_type'].values()]
        }
        split_df = pd.DataFrame(table_data)
        split_df.set_index('Shareholder', inplace=True)
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
    # st.dataframe(cashflow_data)
    st.markdown(html, unsafe_allow_html=True)
    csv = cashflow_data.to_csv(index=True)
    st.download_button(
        label="Download Project Cashflow as CSV",
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


def add_profit_split_tab(summary_calculator, equity_split_data, equity_cashflow_df, cashflow_generator):
    """Add the profit split tab with Tier 1/Tier 2 structure matching the CSV."""
    st.subheader("Profit Split Analysis")
    st.write("Following the **Tier 1 Preferential Entitlement BoP** and **Tier 2** distribution structure")

    # Import the ProfitSplitCalculator
    from cashflow.return_summary import ProfitSplitCalculator

    # Create profit split calculator
    profit_calculator = ProfitSplitCalculator(
        equity_cashflow_df=equity_cashflow_df,
        config=DEFAULT_CONFIG,
        date_processor=cashflow_generator.project_cashflow.date_processor
    )

    # Get total profit from summary calculator
    total_profit = summary_calculator.get_profit()

    # Calculate complete profit distribution
    final_distribution, tier1_results, total_tier1, remaining_profit = profit_calculator.calculate_total_profit_distribution(
        total_profit)

    # Display key parameters
    st.subheader("Key Investment Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Tier 1 Preferential Rates:**")
        st.write("• SAV: 8.5%")
        st.write("• FGC: 8.5%")
        st.write("• FGC2: 12.0%")

    with col2:
        st.write("**Tier 2 Promote:**")
        st.write("• Threshold: 0.0%")
        st.write("• Promote: 25.0%")

    with col3:
        st.write("**Catch-up:**")
        st.write("• Yes")

    # Display summary metrics
    st.subheader("Distribution Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Profit",
            f"£{total_profit:,.0f}",
            help="Net sale price minus total development costs"
        )

    with col2:
        st.metric(
            "Tier 1 Entitlements",
            f"£{total_tier1:,.0f}",
            help="Total preferential entitlements across all shareholders"
        )

    with col3:
        st.metric(
            "Remaining for Tier 2",
            f"£{remaining_profit:,.0f}",
            help="Profit remaining after Tier 1 entitlements"
        )

    with col4:
        tier1_percentage = (total_tier1 / total_profit * 100) if total_profit > 0 else 0
        st.metric(
            "Tier 1 %",
            f"{tier1_percentage:.1f}%",
            help="Percentage of total profit going to Tier 1"
        )

    # Display Tier 1 Preferential Entitlement detailed tables
    st.subheader("Tier 1 Preferential Entitlement BoP")

    # Create tabs for different components
    tier1_tab1, tier1_tab2, tier1_tab3, tier1_tab4 = st.tabs([
        "Beginning of Period", "Monthly Additions", "Monthly Accruals", "End of Period"
    ])

    with tier1_tab1:
        st.write("**Tier 1 Pref Entitlement BoP (Beginning of Period)**")
        bop_display = tier1_results['tier1_bop'].copy()
        for col in bop_display.columns:
            bop_display[col] = bop_display[col].apply(lambda x: f"£{x:,.0f}" if pd.notnull(x) else "£0")
        st.dataframe(bop_display, use_container_width=True)

    with tier1_tab2:
        st.write("**Tier 1 Additions (Monthly Capital Contributions)**")
        additions_display = tier1_results['tier1_additions'].copy()
        for col in additions_display.columns:
            additions_display[col] = additions_display[col].apply(lambda x: f"£{x:,.0f}" if pd.notnull(x) else "£0")
        st.dataframe(additions_display, use_container_width=True)

    with tier1_tab3:
        st.write("**Tier 1 Pref Accrual in Period**")

        # Show rates reference
        st.write("**Quarterly Accrual Rates:**")
        rate_cols = st.columns(3)
        with rate_cols[0]:
            st.write("SAV: 2.125% (8.5% ÷ 4)")
        with rate_cols[1]:
            st.write("FGC: 2.125% (8.5% ÷ 4)")
        with rate_cols[2]:
            st.write("FGC2: 3.0% (12.0% ÷ 4)")

        accruals_display = tier1_results['tier1_accruals'].copy()
        for col in accruals_display.columns:
            accruals_display[col] = accruals_display[col].apply(lambda x: f"£{x:,.0f}" if pd.notnull(x) else "£0")
        st.dataframe(accruals_display, use_container_width=True)

    with tier1_tab4:
        st.write("**Tier 1 Pref Entitlement EoP (End of Period)**")
        eop_display = tier1_results['tier1_eop'].copy()
        for col in eop_display.columns:
            eop_display[col] = eop_display[col].apply(lambda x: f"£{x:,.0f}" if pd.notnull(x) else "£0")
        st.dataframe(eop_display, use_container_width=True)

    # Display final distribution summary
    st.subheader("Final Profit Distribution")

    # Create distribution summary table
    dist_data = []
    for shareholder, values in final_distribution.items():
        clean_name = shareholder.replace("Shareholder Capital: ", "")

        # Determine type (Equity or Loan)
        share_type = "Equity" if shareholder != "Shareholder Capital: FGC2" else "Loan"

        dist_data.append({
            'Type': share_type,
            'Shareholder': clean_name,
            'Tier 1 Rate': f"{values['Tier 1 Rate']:.1f}%",
            'Tier 1 Entitlement': f"£{values['Tier 1 Preferential Entitlement']:,.0f}",
            'Tier 2 Share': f"£{values['Tier 2 Share']:,.0f}",
            'Total Distribution': f"£{values['Total Distribution']:,.0f}",
            'Equity %': f"{values['Equity Percentage']:.1f}%"
        })

    dist_df = pd.DataFrame(dist_data)
    st.dataframe(dist_df, use_container_width=True, hide_index=True)

    # Create waterfall chart showing profit distribution flow
    st.subheader("Profit Distribution Waterfall")

    # Prepare waterfall data
    waterfall_data = {
        'Stage': ['Total Profit', 'SAV Tier 1', 'FGC Tier 1', 'FGC2 Tier 1', 'Remaining for Tier 2', 'SAV Tier 2',
                  'FGC Tier 2'],
        'Amount': [
            total_profit,
            -final_distribution['Shareholder Capital: SAV']['Tier 1 Preferential Entitlement'],
            -final_distribution['Shareholder Capital: FGC']['Tier 1 Preferential Entitlement'],
            -final_distribution['Shareholder Capital: FGC2']['Tier 1 Preferential Entitlement'],
            remaining_profit,
            -final_distribution['Shareholder Capital: SAV']['Tier 2 Share'],
            -final_distribution['Shareholder Capital: FGC']['Tier 2 Share']
        ],
        'Type': ['total', 'relative', 'relative', 'relative', 'total', 'relative', 'relative']
    }

    waterfall_df = pd.DataFrame(waterfall_data)

    # Create waterfall visualization
    fig = px.bar(
        waterfall_df,
        x='Stage',
        y='Amount',
        title='Profit Distribution Waterfall',
        color='Type',
        color_discrete_map={'total': 'blue', 'relative': 'red'}
    )

    fig.update_layout(
        yaxis_tickformat=',.0f',
        yaxis_tickprefix='£',
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Create distribution pie charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Tier 1 Distribution")
        tier1_values = [values['Tier 1 Preferential Entitlement'] for values in final_distribution.values()]
        tier1_labels = [key.replace("Shareholder Capital: ", "") for key in final_distribution.keys()]

        fig1 = px.pie(
            values=tier1_values,
            names=tier1_labels,
            title="Tier 1 Preferential Entitlement",
            hole=0.4
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Tier 2 Distribution")
        tier2_values = [values['Tier 2 Share'] for values in final_distribution.values() if values['Tier 2 Share'] > 0]
        tier2_labels = [key.replace("Shareholder Capital: ", "") for key, values in final_distribution.items() if
                        values['Tier 2 Share'] > 0]

        if tier2_values:
            fig2 = px.pie(
                values=tier2_values,
                names=tier2_labels,
                title="Tier 2 Distribution (Equity Only)",
                hole=0.4
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No Tier 2 distribution (all profit consumed by Tier 1)")

    # Download options
    st.subheader("Download Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        tier1_combined = pd.concat([
            tier1_results['tier1_bop'].add_suffix('_BoP'),
            tier1_results['tier1_additions'].add_suffix('_Add'),
            tier1_results['tier1_accruals'].add_suffix('_Acc'),
            tier1_results['tier1_eop'].add_suffix('_EoP')
        ], axis=1)

        tier1_csv = tier1_combined.to_csv()
        st.download_button(
            label="Download Tier 1 Data",
            data=tier1_csv,
            file_name="tier1_preferential_entitlement.csv",
            mime="text/csv"
        )

    with col2:
        dist_csv = dist_df.to_csv(index=False)
        st.download_button(
            label="Download Distribution Summary",
            data=dist_csv,
            file_name="profit_distribution_summary.csv",
            mime="text/csv"
        )

    with col3:
        # Create detailed breakdown CSV
        detailed_data = []
        for shareholder, values in final_distribution.items():
            detailed_data.append({
                'Shareholder': shareholder,
                'Tier_1_Rate': values['Tier 1 Rate'],
                'Tier_1_Entitlement': values['Tier 1 Preferential Entitlement'],
                'Tier_2_Share': values['Tier 2 Share'],
                'Total_Distribution': values['Total Distribution'],
                'Equity_Percentage': values['Equity Percentage']
            })

        detailed_df = pd.DataFrame(detailed_data)
        detailed_csv = detailed_df.to_csv(index=False)
        st.download_button(
            label="Download Detailed Breakdown",
            data=detailed_csv,
            file_name="detailed_profit_breakdown.csv",
            mime="text/csv"
        )


# Update the create_dashboard function to use the updated profit split tab
def create_dashboard(cashflow_data: pd.DataFrame, cashflow_generator=None,
                     annual_base_rate: float = 0.001, debt_cashflow_df=None, equity_cashflow_df=None,
                     mc_config: dict = None, equity_split_data=None, cash_balance_calculator=None) -> None:
    """Create the Streamlit dashboard with cashflow data and Monte Carlo simulations."""
    st.title('Cashflow Analysis Dashboard')
    summary_calculator = ReturnSummaryCalculator(cashflow_data, DEFAULT_CONFIG)

    # Create tabs for different dashboard sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Project Cashflow",
        "Debt Cashflow",
        "Equity Cashflow",
        "Return Summary",
        "Profit Split",  # Updated tab
        "Monte Carlo Simulation",
        "Profit Analysis"
    ])

    with tab1:
        create_cashflow_analysis(cashflow_data)

    with tab2:
        display_debt_cashflow(debt_cashflow_df, button_key="debt")
        if not cashflow_generator.debt_cashflow_df.empty:
            create_debt_charts(cashflow_generator.debt_cashflow_df)

    with tab3:
        display_debt_cashflow(equity_cashflow_df, button_key='equity')
        if equity_split_data:
            display_equity_split(equity_split_data)

    with tab4:
        add_return_summary_tab(summary_calculator)

    with tab5:  # Updated Profit Split tab
        if equity_cashflow_df is not None and not equity_cashflow_df.empty:
            add_profit_split_tab(summary_calculator, equity_split_data, equity_cashflow_df, cashflow_generator)
        else:
            st.warning("Equity cashflow data not available for profit split analysis.")

    with tab6:
        if mc_config:
            create_monte_carlo_analysis(cashflow_data, mc_config)
        else:
            st.warning("Monte Carlo simulation configuration not provided.")

    with tab7:
        if mc_config and 'profit_details' in mc_config:
            create_profit_analysis(cashflow_data, mc_config['profit_details'])
        else:
            st.warning("Profit details not provided in the configuration.")

