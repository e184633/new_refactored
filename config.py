from datetime import datetime

DEFAULT_CONFIG = {
    # General project configuration
    "acquisition_date": datetime.strptime("Mar-21", "%b-%y"),
    "cutoff_date": datetime.strptime('Dec-24', "%b-%y"),
    "start_date": datetime.strptime("Jan-25", "%b-%y"),
    "end_date": datetime.strptime("May-25", "%b-%y"),
    "refinancing_date": datetime.strptime("Mar-24", "%b-%y"),
    "forecast_periods_count": 5,

    # Development costs configuration
    "additional_unit_cost": 830_000.0,
    "development_cost_adjustment": 0.0,
    "unexpected_costs": 0.0,

    # Base rate configuration
    "annual_base_rate": 0.045,  # Default base rate (4.5%)

    # Input assumptions including financing details
    "input_assumptions": {
        # Development cost assumptions
        "Accountancy": 1455,
        "Legal & Professional": 1640,
        "Management Fee": 39_506,
        "Insurance": 11528,
        "Operation": 12000,
        "Direct Fees": 158_336,
        "Other Project Costs": 780_542,
        "Additional Unit cost": 830_000,

        # Loan facility amounts
        "Senior Facility": 26_000_000,  # Total senior loan facility
        "Mezzanine Facility": 37_000_000,  # Total mezzanine loan facility

        # Loan margin rates (added to base rate for interest calculation)
        "Senior Margin": 0.03,  # 3.0% margin for senior loan
        "Mezzanine Margin": 0.08,  # 8.0% margin for mezzanine loan

        # Non-utilization fee rates
        "Senior Non-utilisation Rate": 0.015,  # 1.5% on undrawn senior amount
        "Mezzanine Non-utilisation Rate": 0.02,  # 2.0% on undrawn mezzanine amount

        # Other fee rates (annual percentage of loan balance)
        "Senior Fees Rate": 0.0025,  # 0.25% annual fee on senior loan
        "Mezzanine Fees Rate": 0.003,  # 0.30% annual fee on mezzanine loan

        # IMS fee rates
        "Senior IMS Rate": 0.0005,  # 0.05% on senior loan
        "Mezzanine IMS Rate": 0.0005,  # 0.05% on mezzanine loan
    },

    # Budget data
    "proposed_budget_data": {
        "pre_antvic_budget": 2_124_436,
        "revised_budget": 21_620_551
    },

    # Monthly base rates (base rate changes over time)
    "annual_base_rates": {
        datetime(2021, 3, 1): 0.0010,  # Mar 2021: 0.10%
        datetime(2021, 4, 1): 0.0010,  # Apr 2021: 0.10%
        datetime(2021, 5, 1): 0.0010,  # May 2021: 0.10%
        datetime(2021, 6, 1): 0.0010,  # Jun 2021: 0.10%
        datetime(2021, 7, 1): 0.0010,  # Jul 2021: 0.10%
        datetime(2021, 8, 1): 0.0010,  # Aug 2021: 0.10%
        datetime(2021, 9, 1): 0.0010,  # Sep 2021: 0.10%
        datetime(2021, 10, 1): 0.0010,  # Oct 2021: 0.10%
        datetime(2021, 11, 1): 0.0010,  # Nov 2021: 0.10%
        datetime(2021, 12, 1): 0.0025,  # Dec 2021: 0.25%
        # 2022: 12 months
        datetime(2022, 1, 1): 0.0025,  # Jan 2022: 0.25%
        datetime(2022, 2, 1): 0.0050,  # Feb 2022: 0.50%
        datetime(2022, 3, 1): 0.0075,  # Mar 2022: 0.75%
        datetime(2022, 4, 1): 0.0075,  # Apr 2022: 0.75%
        datetime(2022, 5, 1): 0.0100,  # May 2022: 1.00%
        datetime(2022, 6, 1): 0.0125,  # Jun 2022: 1.25%
        datetime(2022, 7, 1): 0.0125,  # Jul 2022: 1.25%
        datetime(2022, 8, 1): 0.0175,  # Aug 2022: 1.75%
        datetime(2022, 9, 1): 0.0225,  # Sep 2022: 2.25%
        datetime(2022, 10, 1): 0.0225,  # Oct 2022: 2.25%
        datetime(2022, 11, 1): 0.0300,  # Nov 2022: 3.00%
        datetime(2022, 12, 1): 0.0350,  # Dec 2022: 3.50%
        # 2023: 12 months
        datetime(2023, 1, 1): 0.0350,  # Jan 2023: 3.50%
        datetime(2023, 2, 1): 0.0400,  # Feb 2023: 4.00%
        datetime(2023, 3, 1): 0.0425,  # Mar 2023: 4.25%
        datetime(2023, 4, 1): 0.0425,  # Apr 2023: 4.25%
        datetime(2023, 5, 1): 0.0450,  # May 2023: 4.50%
        datetime(2023, 6, 1): 0.0450,  # Jun 2023: 4.50%
        datetime(2023, 7, 1): 0.0450,  # Jul 2023: 4.50%
        datetime(2023, 8, 1): 0.0500,  # Aug 2023: 5.00%
        datetime(2023, 9, 1): 0.0500,  # Sep 2023: 5.00%
        datetime(2023, 10, 1): 0.0500,  # Oct 2023: 5.00%
        datetime(2023, 11, 1): 0.0525,  # Nov 2023: 5.25%
        datetime(2023, 12, 1): 0.0525,  # Dec 2023: 5.25%
        # 2024: 12 months
        datetime(2024, 1, 1): 0.0525,  # Jan 2024: 5.25%
        datetime(2024, 2, 1): 0.0525,  # Feb 2024: 5.25%
        datetime(2024, 3, 1): 0.0525,  # Mar 2024: 5.25%
        datetime(2024, 4, 1): 0.0525,  # Apr 2024: 5.25%
        datetime(2024, 5, 1): 0.0525,  # May 2024: 5.25%
        datetime(2024, 6, 1): 0.0525,  # Jun 2024: 5.25%
        datetime(2024, 7, 1): 0.0525,  # Jul 2024: 5.25%
        datetime(2024, 8, 1): 0.0525,  # Aug 2024: 5.25%
        datetime(2024, 9, 1): 0.0525,  # Sep 2024: 5.25%
        datetime(2024, 10, 1): 0.0525,  # Oct 2024: 5.25%
        datetime(2024, 11, 1): 0.0500,  # Nov 2024: 5.00%
        datetime(2024, 12, 1): 0.0500,  # Dec 2024: 5.00%
        # 2025: January to May (5 months)
        datetime(2025, 1, 1): 0.0500,  # Jan 2025: 5.00%
        datetime(2025, 2, 1): 0.0475,  # Feb 2025: 4.75%
        datetime(2025, 3, 1): 0.0475,  # Mar 2025: 4.75%
        datetime(2025, 4, 1): 0.0475,  # Apr 2025: 4.75%
        datetime(2025, 5, 1): 0.0450,  # May 2025: 4.50%
    },

    # Monte Carlo simulation parameters (if needed)
    "monte_carlo": {
        "num_simulations": 1000,
        "exit_value": {
            "mean": 79_500_000,
            "min": 75_000_000,
            "max": 84_000_000
        },
        "cost_variation": {
            "acquisition": {
                "min": 0.01,
                "max": 0.05
            },
            "development": {
                "min": 0.03,
                "max": 0.15
            }
        },
        "project_timeline": {
            "acquisition": 0.15,
            "early_development": 0.35,
            "mid_development": 0.35,
            "late_development": 0.15
        },
        "profit_details": {
            "sell_price": 79_500_000,
            "selling_cost_percentage": .002,
            "corp_tax_percentage": .25,
        }
    }
}

# Categories for organizing the data
CATEGORIES = {
    "all": [
        "Actual/Forecast", "Acquisition", "SDLT", "Agent Fees", "Development", "Accountancy",
        "Planning & Design", "Legal & Professional", "Management Fee", "Insurance",
        "VAT", "Additional", "Operation", "Total Development Costs", "Total Project Cashflow", "Acquisition Total",
        "Financing costs",
    ],
    "development": [
        "Accountancy", "Planning & Design", "Development costs", "Legal & Professional",
        "Management Fee", "Insurance", "Operation", "VAT", "Additional", "Financing costs"
    ],
    "historical": [
        "Acquisition", "SDLT", "Agent Fees", "Accountancy", "Planning & Design",
        "Development costs", "Legal & Professional", "Management Fee", "Insurance",
        "VAT", "Additional", "Operation"
    ],
    "development_components": [
        "Accountancy", "Planning & Design", "Development costs", "Legal & Professional",
        "Management Fee", "Insurance", "Operation", "VAT", "Additional",
    ],
    "acquisition_components": [
        "Acquisition",
        "SDLT",
        "Agent Fees",
    ],
    'senior_loan': [
        "Acquisition", "Fees", "Development", "Capitalised Interest",
        "Non-utilisation Fee", "IMS Fees", "Cash Payment", "Loan Redemption"
    ],
    'mezzanine_loan': [
        "Refinancing", "Fees", "Development", "Capitalised Interest",
        "Non-utilisation Fee", "IMS Fees", "Cash Payment"
    ],
    "financing": ["Financing costs"],
    "total_project_components": ["Acquisition", "Development", "Financing costs"],
    "quarterly": ["Management Fee", "Accountancy"],
}