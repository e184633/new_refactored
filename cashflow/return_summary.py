# Updated cashflow/return_summary.py to match the CSV structure

import pandas as pd


class ReturnSummaryCalculator:
    """Centralizes calculations for financial summaries and returns."""

    def __init__(self, cashflow_df, config):
        self.cashflow_df = cashflow_df
        self.config = config

        # Calculate and store key values once
        self._calculate_values()

    def _calculate_values(self):
        """Calculate and store all key financial metrics."""
        # Exit value calculations
        self.exit_amount = self.config.get('input_assumptions', {}).get('exit_amount', 79500000)
        self.selling_costs_pct = self.config.get('selling_costs_pct', 0.002)
        self.selling_costs = -self.exit_amount * self.selling_costs_pct
        self.net_sale_price = self.exit_amount + self.selling_costs

        # Cost calculations
        self.acquisition_total = 0
        self.development_costs = 0
        self.financing_costs = 0
        self.total_development_cost = 0

        if self.cashflow_df is not None and not self.cashflow_df.empty:
            for idx, row in self.cashflow_df.iterrows():
                category = row.get('Category', row.name)
                if category == 'Acquisition Total' and 'Total' in self.cashflow_df.columns:
                    self.acquisition_total = row['Total']
                elif category == 'Development' and 'Total' in self.cashflow_df.columns:
                    self.development_costs = row['Total']
                elif category == 'Financing costs' and 'Total' in self.cashflow_df.columns:
                    self.financing_costs = row['Total']
                elif category == 'Total Development Costs' and 'Total' in self.cashflow_df.columns:
                    self.total_development_cost = row['Total']

    def get_exit_value_df(self):
        """Return a DataFrame for the Exit Value table."""
        exit_data = {
            "Item": ["Sell price", "Selling Costs", "Net sale price"],
            "Value": [
                f"£{self.exit_amount:,.0f}",
                f"(£{abs(self.selling_costs):,.0f})",
                f"£{self.net_sale_price:,.0f}"
            ]
        }
        return pd.DataFrame(exit_data).set_index("Item")

    def get_costs_df(self):
        """Return a DataFrame for the Costs table."""
        costs_data = {
            "Item": ["Acquisition", "Development", "Financing costs", "Total Development cost"],
            "Value": [
                f"(£{abs(self.acquisition_total):,.0f})" if self.acquisition_total < 0 else f"£{self.acquisition_total:,.0f}",
                f"(£{abs(self.development_costs):,.0f})" if self.development_costs < 0 else f"£{self.development_costs:,.0f}",
                f"(£{abs(self.financing_costs):,.0f})" if self.financing_costs < 0 else f"£{self.financing_costs:,.0f}",
                f"(£{abs(self.total_development_cost):,.0f})" if self.total_development_cost < 0 else f"£{self.total_development_cost:,.0f}"
            ]
        }
        return pd.DataFrame(costs_data).set_index("Item")

    def get_profit(self):
        """Calculate profit (net sale price minus total development cost)."""
        return self.net_sale_price + self.total_development_cost  # Adding a negative number


class ProfitSplitCalculator:
    """Calculates Tier 1 preferential entitlement BoP following the CSV structure."""

    def __init__(self, equity_cashflow_df, config, date_processor):
        self.equity_cashflow_df = equity_cashflow_df
        self.config = config
        self.date_processor = date_processor
        self.shareholder_names = config['shareholder_names']
        self.financing_rates = config['financing_type']

        # Map shareholders to their Tier 1 rates (matching CSV structure)
        self.tier1_rates = {
            "Shareholder Capital: SAV": 0.085,  # 8.5%
            "Shareholder Capital: FGC": 0.085,  # 8.5%
            "Shareholder Capital: FGC2": 0.120  # 12.0%
        }

        # Tier 2 promote rates
        self.tier1_threshold = 0.0  # 0% threshold for Tier 1
        self.tier2_promote = 0.25  # 25% promote for Tier 2

    def calculate_tier1_preferential_entitlement(self):
        """Calculate Tier 1 preferential entitlement BoP for each shareholder by month."""
        monthly_periods = self.date_processor.monthly_period_strs

        # Initialize result dataframes following CSV structure
        columns = ['BoP Start'] + monthly_periods + ['Cumulative Entitlement']

        # Create separate dataframes for each component (matching CSV structure)
        bop_df = pd.DataFrame(index=self.shareholder_names + ['Total'], columns=columns)
        addition_df = pd.DataFrame(index=self.shareholder_names + ['Total'],
                                   columns=monthly_periods + ['Total Additions'])
        accrual_df = pd.DataFrame(index=self.shareholder_names + ['Total'],
                                  columns=monthly_periods + ['Total Accruals'])
        eop_df = pd.DataFrame(index=self.shareholder_names + ['Total'], columns=columns)

        # Initialize all values to 0
        bop_df = bop_df.fillna(0.0)
        addition_df = addition_df.fillna(0.0)
        accrual_df = accrual_df.fillna(0.0)
        eop_df = eop_df.fillna(0.0)

        # Get shareholder contributions by period
        shareholder_contributions = {}
        for shareholder in self.shareholder_names:
            contributions = {}
            for period in monthly_periods:
                shareholder_row = self.equity_cashflow_df[
                    self.equity_cashflow_df['Category'] == shareholder
                    ]
                if not shareholder_row.empty and period in shareholder_row.columns:
                    contributions[period] = float(shareholder_row[period].iloc[0])
                else:
                    contributions[period] = 0.0
            shareholder_contributions[shareholder] = contributions

        # Calculate BoP and accruals for each shareholder by month
        for shareholder in self.shareholder_names:
            rate = self.tier1_rates[shareholder]
            monthly_rate = rate / 12  # Convert annual rate to monthly

            running_balance = 0.0
            cumulative_accruals = 0.0

            # Set starting BoP
            bop_df.loc[shareholder, 'BoP Start'] = running_balance

            for period in monthly_periods:
                # Beginning of Period balance
                bop_df.loc[shareholder, period] = running_balance

                # Calculate additions (capital contributions) for this month
                month_addition = shareholder_contributions[shareholder][period]
                addition_df.loc[shareholder, period] = month_addition

                # Add contributions to running balance
                running_balance += month_addition

                # Calculate preferential accrual on the balance (including new additions)
                monthly_accrual = running_balance * monthly_rate
                accrual_df.loc[shareholder, period] = monthly_accrual
                cumulative_accruals += monthly_accrual

                # End of Period balance (BoP + additions, accrual tracked separately)
                eop_df.loc[shareholder, period] = running_balance

            # Store cumulative figures
            bop_df.loc[shareholder, 'Cumulative Entitlement'] = cumulative_accruals
            addition_df.loc[shareholder, 'Total Additions'] = sum(shareholder_contributions[shareholder].values())
            accrual_df.loc[shareholder, 'Total Accruals'] = cumulative_accruals
            eop_df.loc[shareholder, 'Cumulative Entitlement'] = running_balance + cumulative_accruals

        # Calculate totals across all shareholders
        for col in bop_df.columns:
            if col != 'Cumulative Entitlement':
                bop_df.loc['Total', col] = bop_df.loc[self.shareholder_names, col].sum()

        for col in addition_df.columns:
            if col != 'Total Additions':
                addition_df.loc['Total', col] = addition_df.loc[self.shareholder_names, col].sum()

        for col in accrual_df.columns:
            if col != 'Total Accruals':
                accrual_df.loc['Total', col] = accrual_df.loc[self.shareholder_names, col].sum()

        for col in eop_df.columns:
            if col != 'Cumulative Entitlement':
                eop_df.loc['Total', col] = eop_df.loc[self.shareholder_names, col].sum()

        # Calculate total cumulative values
        bop_df.loc['Total', 'Cumulative Entitlement'] = bop_df.loc[
            self.shareholder_names, 'Cumulative Entitlement'].sum()
        addition_df.loc['Total', 'Total Additions'] = addition_df.loc[self.shareholder_names, 'Total Additions'].sum()
        accrual_df.loc['Total', 'Total Accruals'] = accrual_df.loc[self.shareholder_names, 'Total Accruals'].sum()
        eop_df.loc['Total', 'Cumulative Entitlement'] = eop_df.loc[
            self.shareholder_names, 'Cumulative Entitlement'].sum()

        return {
            'tier1_bop': bop_df,
            'tier1_additions': addition_df,
            'tier1_accruals': accrual_df,
            'tier1_eop': eop_df
        }

    def calculate_tier2_distribution(self, total_profit, tier1_results):
        """Calculate Tier 2 distribution after Tier 1 entitlements."""
        total_tier1_entitlement = tier1_results['tier1_accruals'].loc['Total', 'Total Accruals']

        # Remaining profit after Tier 1 preferential entitlement
        remaining_profit = total_profit - total_tier1_entitlement

        if remaining_profit <= 0:
            return None, 0, remaining_profit

        # Get equity ownership percentages for Tier 2 distribution
        equity_split = {}
        total_equity = 0

        for shareholder in self.shareholder_names:
            shareholder_row = self.equity_cashflow_df[
                self.equity_cashflow_df['Category'] == shareholder
                ]
            if not shareholder_row.empty and 'Total' in shareholder_row.columns:
                value = float(shareholder_row['Total'].iloc[0])
                if shareholder != "Shareholder Capital: FGC2":  # FGC2 is loan, not equity for Tier 2
                    equity_split[shareholder] = value
                    total_equity += value

        # Calculate Tier 2 distribution (only among equity holders)
        tier2_distribution = {}
        for shareholder in equity_split:
            if total_equity > 0:
                ownership_pct = equity_split[shareholder] / total_equity
                tier2_share = remaining_profit * ownership_pct
                tier2_distribution[shareholder] = tier2_share
            else:
                tier2_distribution[shareholder] = 0

        return tier2_distribution, total_tier1_entitlement, remaining_profit

    def calculate_total_profit_distribution(self, total_profit):
        """Calculate complete profit distribution including Tier 1 and Tier 2."""
        # Calculate Tier 1 preferential entitlements
        tier1_results = self.calculate_tier1_preferential_entitlement()

        # Calculate Tier 2 distribution
        tier2_distribution, total_tier1, remaining_profit = self.calculate_tier2_distribution(
            total_profit, tier1_results
        )

        # Combine Tier 1 and Tier 2 for final distribution
        final_distribution = {}

        for shareholder in self.shareholder_names:
            tier1_entitlement = tier1_results['tier1_accruals'].loc[shareholder, 'Total Accruals']
            tier2_share = tier2_distribution.get(shareholder, 0) if tier2_distribution else 0

            # Get equity percentage
            shareholder_row = self.equity_cashflow_df[
                self.equity_cashflow_df['Category'] == shareholder
                ]
            equity_value = 0
            if not shareholder_row.empty and 'Total' in shareholder_row.columns:
                equity_value = float(shareholder_row['Total'].iloc[0])

            total_equity = sum(
                float(row['Total']) for _, row in self.equity_cashflow_df.iterrows()
                if row['Category'] in self.shareholder_names and 'Total' in self.equity_cashflow_df.columns
            )

            equity_percentage = (equity_value / total_equity * 100) if total_equity > 0 else 0

            final_distribution[shareholder] = {
                'Tier 1 Preferential Entitlement': tier1_entitlement,
                'Tier 2 Share': tier2_share,
                'Total Distribution': tier1_entitlement + tier2_share,
                'Equity Percentage': equity_percentage,
                'Tier 1 Rate': self.tier1_rates[shareholder] * 100  # Convert to percentage
            }

        return final_distribution, tier1_results, total_tier1, remaining_profit


# Example usage and configuration setup
def setup_default_profit_split_config():
    """
    Example of how to set up the profit split configuration.
    """

    default_config = {
        'shareholder_names': [
            'Shareholder Capital: SAV',
            'Shareholder Capital: FGC',
            'Shareholder Capital: FGC2'
        ],
        'profit_split': {
            'tiers': {
                'tier_1': {
                    'threshold_percentage': 0.0,
                    'description': 'Preferential Return',
                    'calculation_method': 'preferential_return'
                },
                'tier_2': {
                    'threshold_percentage': 0.0,
                    'description': 'Return of Capital',
                    'calculation_method': 'return_of_capital'
                },
                'tier_3': {
                    'threshold_percentage': 0.0,
                    'description': 'Promote Distribution',
                    'calculation_method': 'promote_split'
                }
            },
            'shareholder_rates': {
                'Shareholder Capital: SAV': {
                    'tier_1_rate': 0.085,  # 8.5%
                    'tier_2_ownership': None,
                    'tier_3_promote': 0.0
                },
                'Shareholder Capital: FGC': {
                    'tier_1_rate': 0.085,  # 8.5%
                    'tier_2_ownership': None,
                    'tier_3_promote': 0.0
                },
                'Shareholder Capital: FGC2': {
                    'tier_1_rate': 0.120,  # 12.0%
                    'tier_2_ownership': None,
                    'tier_3_promote': 0.0
                }
            },
            'promote_structure': {
                'catch_up_threshold': 0.0,
                'promote_percentage': 0.25,  # 25%
                'promote_recipient': 'Shareholder Capital: FGC',  # or 'General Partner'
                'remaining_split': 'pro_rata'
            }
        }
    }

    return default_config
