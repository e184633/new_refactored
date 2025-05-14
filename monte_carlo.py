from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import numpy_financial as npf
import pandas as pd
import plotly.express as px
from dateutil.relativedelta import relativedelta
from pyxirr import xirr

from config import DEFAULT_CONFIG

SAV_DATE_FORMAT = '%b-%y'

OVERVIEW_COLUMNS = [f"Inception to {DEFAULT_CONFIG['cutoff_date'].strftime(SAV_DATE_FORMAT)}",
                    f"{DEFAULT_CONFIG['start_date'].strftime(SAV_DATE_FORMAT)} to Exit", 'Total']

def calculate_irr(dates: List[datetime], cashflows: List[float]) -> float:
    """
    Calculate the Internal Rate of Return (IRR) for a series of cashflows with dates.

    Args:
        dates: List of dates for each cashflow
        cashflows: List of cash flows, starting with the initial investment (negative value)

    Returns:
        The IRR as a decimal (not percentage)
    """
    # Handle the case where all cashflows are 0 or there's only one cashflow
    if all(cf == 0 for cf in cashflows) or len(cashflows) <= 1:
        return 0.0

    try:
        return xirr(dates, cashflows)
    except NameError:
        # Fallback to numpy if pyxirr is not available
        return npf.irr(cashflows)


def calculate_profit_on_cost(total_costs: float, exit_value: float) -> float:
    """
    Calculate Profit on Cost (POC).

    Args:
        total_costs: Total project costs (Acquisition + Development + Financing)
        exit_value: Exit value (sale price)

    Returns:
        Profit on Cost as a decimal (not percentage)
    """
    if total_costs == 0:
        return 0.0

    profit = exit_value - total_costs
    return profit / total_costs


def run_monte_carlo_simulation(
        cashflow_df: pd.DataFrame,
        num_simulations: int,
        exit_value_params: Dict[str, float],
        cost_variation_params: Dict[str, Dict[str, float]],
        project_timeline: Dict[str, float] = None
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Run Monte Carlo simulation on the cashflow model, focusing on IRR and Profit on Cost.

    Args:
        cashflow_df: Cashflow DataFrame with periods as columns
        num_simulations: Number of Monte Carlo simulations to run
        exit_value_params: Dict with {'mean', 'min', 'max'} for exit value triangular distribution
        cost_variation_params: Dict with parameters for cost variations
        financing_cost_params: Dict with financing cost parameters

    Returns:
        Tuple containing (results_df, summary_stats)
        :param cost_variation_params:
        :param num_simulations:
        :param project_timeline:
    """
    # Extract necessary data from cashflow_df
    # Only use monthly columns (not the summary columns)
    period_columns = [col for col in cashflow_df.columns if col not in OVERVIEW_COLUMNS]
    print(period_columns, OVERVIEW_COLUMNS)
    # Parse period strings to dates for IRR calculation
    period_dates = []
    for period_str in period_columns:
        date = pd.to_datetime(period_str, format=SAV_DATE_FORMAT)
        period_dates.append(date)
    print(len(period_dates))
    # Set up default project timeline if not provided
    if project_timeline is None:
        project_timeline = {
            'acquisition': 0.15,  # 15% at the beginning
            'early_development': 0.35,  # 35% in early stages
            'mid_development': 0.35,  # 35% in middle stages
            'late_development': 0.15  # 15% near completion
        }

    # Create arrays to store simulation results
    irr_results = np.zeros(num_simulations)
    poc_results = np.zeros(num_simulations)
    exit_values = np.zeros(num_simulations)
    acquisition_costs = np.zeros(num_simulations)
    development_costs = np.zeros(num_simulations)
    total_costs = np.zeros(num_simulations)

    # Run simulations
    for i in range(num_simulations):
        # Simulate exit value using triangular distribution
        exit_value = np.random.triangular(
            exit_value_params['min'],
            exit_value_params['mean'],
            exit_value_params['max']
        )
        exit_values[i] = exit_value

        # Get baseline costs
        base_acquisition_cost = abs(cashflow_df.loc['Acquisition Total', 'Total'])
        base_development_cost = abs(cashflow_df.loc['Total Development Costs', 'Total'])

        # Simulate acquisition cost variation
        acq_var = cost_variation_params['acquisition']
        acquisition_cost = base_acquisition_cost * np.random.uniform(
            1 - acq_var['min'],
            1 + acq_var['max']
        )
        acquisition_costs[i] = acquisition_cost

        # Simulate development cost variation
        dev_var = cost_variation_params['development']
        development_cost = base_development_cost * np.random.uniform(
            1 - dev_var['min'],
            1 + dev_var['max']
        )
        development_costs[i] = development_cost

        # Calculate total costs without financing
        total_cost = acquisition_cost + development_cost
        total_costs[i] = total_cost

        # Simulate cashflow for IRR calculation with realistic timing
        cashflow_values = []
        cashflow_dates = []

        # First date (acquisition date - typically the first period)
        if period_dates:
            start_date = period_dates[0]
            cashflow_dates.append(start_date)
            # Initial outflow (acquisition costs)
            cashflow_values.append(-acquisition_cost)

            # Distribute development costs according to project timeline
            num_periods = len(period_dates)

            # Determine phase boundaries
            early_bound = int(num_periods * 0.25)  # First 25% of periods
            mid_bound = int(num_periods * 0.75)  # Middle 50% of periods

            # Development costs spread with realistic timing
            for j, date in enumerate(period_dates[1:], 1):  # Skip the first period (acquisition)
                cashflow_dates.append(date)

                # Determine which phase we're in and apply the appropriate cost percentage
                if j < early_bound:
                    # Early development phase - higher costs for foundation and initial work
                    period_cost = -(development_cost * project_timeline['early_development']) / early_bound
                elif j < mid_bound:
                    # Mid development phase - steady construction costs
                    period_cost = -(development_cost * project_timeline['mid_development']) / (mid_bound - early_bound)
                else:
                    # Late development phase - finishing work, typically lower costs
                    period_cost = -(development_cost * project_timeline['late_development']) / (num_periods - mid_bound)

                # The last period also includes the exit value
                if j == len(period_dates) - 1:
                    period_cost += exit_value

                cashflow_values.append(period_cost)

        # Calculate IRR using dates and cashflows
        irr = calculate_irr(cashflow_dates, cashflow_values)
        irr_results[i] = irr

        # Calculate POC
        poc = calculate_profit_on_cost(total_cost, exit_value)
        irr_results[i] = irr

        poc = calculate_profit_on_cost(total_cost, exit_value)
        poc_results[i] = poc

    # Create results DataFrame
    results_df = pd.DataFrame({
        'IRR': irr_results,
        'POC': poc_results,
        'Exit Value': exit_values,
        'Acquisition Costs': acquisition_costs,
        'Development Costs': development_costs,
        'Total Costs': total_costs
    })

    # Calculate summary statistics
    summary_stats = {
        'IRR': {
            'mean': results_df['IRR'].mean(),
            'median': results_df['IRR'].median(),
            'min': results_df['IRR'].min(),
            'max': results_df['IRR'].max(),
            'std': results_df['IRR'].std(),
            'percentiles': {
                5: np.percentile(results_df['IRR'], 5),
                25: np.percentile(results_df['IRR'], 25),
                75: np.percentile(results_df['IRR'], 75),
                95: np.percentile(results_df['IRR'], 95)
            }
        },
        'POC': {
            'mean': results_df['POC'].mean(),
            'median': results_df['POC'].median(),
            'min': results_df['POC'].min(),
            'max': results_df['POC'].max(),
            'std': results_df['POC'].std(),
            'percentiles': {
                5: np.percentile(results_df['POC'], 5),
                25: np.percentile(results_df['POC'], 25),
                75: np.percentile(results_df['POC'], 75),
                95: np.percentile(results_df['POC'], 95)
            }
        }
    }

    return results_df, summary_stats


def create_simulation_plots(results_df: pd.DataFrame, summary_stats: Dict[str, Dict[str, float]]):
    """
    Create visualization plots for Monte Carlo simulation results.

    Args:
        results_df: DataFrame with simulation results
        summary_stats: Dictionary with summary statistics

    Returns:
        Dictionary of Plotly figures
    """
    plots = {}

    # IRR Distribution
    irr_fig = px.histogram(
        results_df, x='IRR',
        title='IRR Distribution',
        labels={'IRR': 'Internal Rate of Return'},
        marginal='box',
        nbins=50,
        color_discrete_sequence=['#1f77b4']
    )

    # Add median line
    irr_fig.add_vline(
        x=summary_stats['IRR']['median'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Median: {summary_stats['IRR']['median']:.2%}",
        annotation_position="top right"
    )

    # Add 25th and 75th percentile lines
    irr_fig.add_vline(
        x=summary_stats['IRR']['percentiles'][25],
        line_dash="dot",
        line_color="orange",
        annotation_text=f"25th: {summary_stats['IRR']['percentiles'][25]:.2%}",
        annotation_position="top left"
    )

    irr_fig.add_vline(
        x=summary_stats['IRR']['percentiles'][75],
        line_dash="dot",
        line_color="orange",
        annotation_text=f"75th: {summary_stats['IRR']['percentiles'][75]:.2%}",
        annotation_position="top right"
    )

    irr_fig.update_xaxes(tickformat='.1%')
    plots['irr_histogram'] = irr_fig

    # POC Distribution
    poc_fig = px.histogram(
        results_df, x='POC',
        title='Profit on Cost Distribution',
        labels={'POC': 'Profit on Cost'},
        marginal='box',
        nbins=50,
        color_discrete_sequence=['#2ca02c']
    )

    # Add median line
    poc_fig.add_vline(
        x=summary_stats['POC']['median'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Median: {summary_stats['POC']['median']:.2%}",
        annotation_position="top right"
    )

    # Add 25th and 75th percentile lines
    poc_fig.add_vline(
        x=summary_stats['POC']['percentiles'][25],
        line_dash="dot",
        line_color="orange",
        annotation_text=f"25th: {summary_stats['POC']['percentiles'][25]:.2%}",
        annotation_position="top left"
    )

    poc_fig.add_vline(
        x=summary_stats['POC']['percentiles'][75],
        line_dash="dot",
        line_color="orange",
        annotation_text=f"75th: {summary_stats['POC']['percentiles'][75]:.2%}",
        annotation_position="top right"
    )

    poc_fig.update_xaxes(tickformat='.1%')
    plots['poc_histogram'] = poc_fig

    # Scatter plot of IRR vs POC
    scatter_fig = px.scatter(
        results_df, x='IRR', y='POC',
        title='IRR vs Profit on Cost',
        labels={'IRR': 'Internal Rate of Return', 'POC': 'Profit on Cost'},
        opacity=0.6,
        color='Exit Value',
        color_continuous_scale='Viridis'
    )
    scatter_fig.update_xaxes(tickformat='.1%')
    scatter_fig.update_yaxes(tickformat='.1%')
    plots['irr_poc_scatter'] = scatter_fig

    # Box plots for cost components
    cost_df = results_df[['Acquisition Costs', 'Development Costs']].melt(var_name='Cost Type',
                                                                          value_name='Amount')
    cost_box_fig = px.box(
        cost_df, x='Cost Type', y='Amount',
        title='Distribution of Cost Components',
        color='Cost Type',
        points='all'
    )
    plots['cost_boxplot'] = cost_box_fig

    return plots
