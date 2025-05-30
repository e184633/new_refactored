import copy
import json

import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta

from config import DEFAULT_CONFIG
from cashflow import CashflowGenerator
from dashboard import create_dashboard
from datetime import datetime

SAV_DATE_FORMAT = '%b-%y'
FORECAST_START_DATE = DEFAULT_CONFIG['cutoff_date'] + relativedelta(months=1)
OVERVIEW_COLUMNS = [f"Inception to {DEFAULT_CONFIG['cutoff_date'].strftime(SAV_DATE_FORMAT)}",
                    f"{DEFAULT_CONFIG['start_date'].strftime(SAV_DATE_FORMAT)} to Exit", 'Total']
ACQUISITION_DATE = DEFAULT_CONFIG['acquisition_date']
CUTOFF_DATE = DEFAULT_CONFIG.get('cutoff_date') + relativedelta(day=31)
EXIT_DATE = DEFAULT_CONFIG.get('exit_date') + relativedelta(day=31)
DEVELOPMENT_COSTS_STR = 'Development costs'
ANTVIC_STR = 'antvic'
QUARTERLY_MONTH_INDEX = 2  # Third month in quarter (0-based index)


# Set the page layout to wide (must be the first Streamlit command)
st.set_page_config(layout="wide")
# After st.set_page_config() at the top of your app
STATCOUNTER_CODE = """
<!-- Default Statcounter code for Master Model -->
<script type="text/javascript">
var sc_project=13135593; 
var sc_invisible=1; 
var sc_security="ff856198"; 
</script>
<script type="text/javascript"
src="https://www.statcounter.com/counter/counter.js" async></script>
<noscript><div class="statcounter"><a title="Web Analytics Made Easy -
Statcounter" href="https://statcounter.com/" target="_blank"><img
class="statcounter" src="https://c.statcounter.com/13135593/0/ff856198/1/"
alt="Web Analytics Made Easy - Statcounter"
referrerPolicy="no-referrer-when-downgrade"></a></div></noscript>
<!-- End of Statcounter Code -->
"""

st.markdown(STATCOUNTER_CODE, unsafe_allow_html=True)

# Force light theme with a single line of CSS
st.markdown('<style>.stApp {background-color: #FFFFFF; color: #31333F;}</style>', unsafe_allow_html=True)


# Main Application
@st.cache_data
def load_bank_statement(file_path):
    """ Load the bank statement Excel file with caching. """
    bank_statement = pd.read_excel(
        file_path,
        sheet_name='More House Bank',
        # skiprows=1,
        usecols={'SOURCE', 'TAG', 'MOVEMENT', 'MONTH', 'DETAIL 1', 'DETAIL 2', 'CASH IN', 'CASH OUT'}
    )
    bank_statement = bank_statement[~((bank_statement.isna()) | (bank_statement == 'x')).all(axis=1)]
    bank_statement['MONTH'] = pd.to_datetime(bank_statement['MONTH'])
    return bank_statement.sort_values('MONTH', ascending=True)


def create_input_form(default_config: dict) -> dict:
    """ Create the input form in the sidebar and return user inputs. """
    st.sidebar.header("Input Parameters")

    # Date Parameters
    with st.sidebar.expander("Date Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            # Use date_input instead of text_input
            acquisition_date = st.date_input(
                "Acquisition Date",
                value=default_config['acquisition_date'],
                min_value=datetime(2000, 1, 1),  # Optional: set minimum date
                max_value=datetime(2030, 12, 31),  # Optional: set maximum date
                key="acq_date",
                # format="MMM-YY"  # Displays as "Dec-24"
            )
            # print(acquisition_date, type(acquisition_date))
        with col2:
            start_date = st.date_input(
                "Start date",
                value=default_config["start_date"],
                min_value=datetime(2000, 1, 1),  # Optional: set minimum date
                max_value=datetime(2030, 12, 31),  # Optional: set maximum date
                key="start_date")
            # print(start_date, type(start_date))
        col3, col4 = st.columns(2)
        with col3:
            exit_date = st.date_input(
                "Exit Date",
                min_value=datetime(2000, 1, 1),  # Optional: set minimum date
                max_value=datetime(2030, 12, 31),  # Optional: set maximum date
                value=default_config["exit_date"],
                key="exit_date")
        with col4:
            forecast_periods_count = st.number_input(
                "Periods",
                min_value=1,
                value=default_config["forecast_periods_count"],
                step=1,
                key="periods")
        col5, col6 = st.columns(2)
        with col3:
            start_of_cash_payment = st.date_input(
                "Cash Payments Start Date",
                min_value=datetime(2000, 1, 1),  # Optional: set minimum date
                max_value=datetime(2030, 12, 31),  # Optional: set maximum date
                value=default_config["start_of_cash_payment"],
                key="start_of_cash_payment")
        with col4:
            construction_end_date = st.date_input(
                "Construction End Date",
                min_value=datetime(2000, 1, 1),  # Optional: set minimum date
                max_value=datetime(2030, 12, 31),  # Optional: set maximum date
                value=default_config["construction_end_date"],
                key="construction_end_date")

    # Input Assumptions
    with st.sidebar.expander("Input Assumptions (£)"):
        input_assumptions = {}
        for key, default_value in default_config["input_assumptions"].items():
            input_assumptions[key] = st.number_input(
                key.replace(' & ', ' ').replace('Additional Unit cost', 'Add. Unit Cost'),
                min_value=0.0,
                value=float(default_value),
                step=100.0, format="%.0f",
                key=f"assump_{key}"
            )

    # Proposed Budget Data
    with st.sidebar.expander("Proposed Budget (£)"):
        proposed_budget_data = {}
        for key, default_value in default_config["proposed_budget_data"].items():
            label = "Pre-Antvic Budget" if key == 'pre_antvic_budget' else "Revised Budget"
            value = st.number_input(
                label,
                min_value=0.0,
                value=float(default_value),
                step=1000.0,
                format="%.0f",
                key=f"budget_{key}"
            )
            proposed_budget_data[key] = value
            st.text(f"Formatted: £{value:,.0f}")

    # Additional Parameters
    with st.sidebar.expander("Additional Parameters"):
        col5, col6 = st.columns(2)
        with col5:
            additional_unit_cost = st.number_input(
                "Add. Unit Cost (£)",
                min_value=0.0,
                value=default_config["additional_unit_cost"],
                step=1000.0, format="%.0f",
                key="add_unit_cost"
            )
        with col6:
            development_cost_adjustment = st.number_input(
                "Dev. Cost Adj. (£)",
                value=default_config["development_cost_adjustment"],
                step=100.0,
                format="%.0f",
                key="dev_cost_adj"
            )
        col7, col8 = st.columns(2)
        with col7:
            annual_base_rate = st.number_input(
                "Base Rate",
                min_value=0.0,
                value=default_config["annual_base_rate"],
                step=0.0001,
                format="%.4f",
                key="base_rate "
            )
        with col8:
            unexpected_costs = st.number_input(
                "Unexpected Costs (£)",
                min_value=0.0,
                value=default_config["unexpected_costs"],
                step=1000.0,
                format="%.0f",
                key="unexpected_costs"
            )

    # Financing Types Section
    with st.sidebar.expander("Financing Types"):
        # Create individual fields for each financing type
        equity_rate = st.number_input(
            "Equity Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.0850,  # Hard-coded default
            step=0.001,
            format="%.4f",
            key="financing_equity"
        )
        st.text(f"Formatted: {equity_rate:.2%}")

        loan_rate = st.number_input(
            "Loan Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.1200,  # Hard-coded default
            step=0.001,
            format="%.4f",
            key="financing_loan"
        )
        st.text(f"Formatted: {loan_rate:.2%}")

        # Manually create the financing_type dictionary
        financing_types = {
            "Equity": equity_rate,
            "Loan": loan_rate
        }

        # # Add exit amount parameter
        # exit_amount = st.number_input(
        #     "Exit Amount (£)",
        #     min_value=0.0,
        #     value=79_500_000,  # Hard-coded default
        #     step=100000.0,
        #     format="%.0f",
        #     key="exit_amount"
        # )
        # st.text(f"Formatted: £{exit_amount:,.0f}")
    return {
        "acquisition_date": acquisition_date,
        "start_date": start_date,
        "end_date": exit_date,
        "start_of_cash_payment": start_of_cash_payment,
        "construction_end_date": construction_end_date,
        "forecast_periods_count": forecast_periods_count,
        "input_assumptions": input_assumptions,
        # "financing_type": financing_types,  # Add the financing types
        # "exit_amount": exit_amount,  # Add exit amount
        "proposed_budget_data": proposed_budget_data,
        "additional_unit_cost": additional_unit_cost,
        "development_cost_adjustment": development_cost_adjustment,
        "annual_base_rate": annual_base_rate,
        "unexpected_costs": unexpected_costs
    }


def load_loan_statement(file_path: str, loan_name: str):
    bank_statement = load_bank_statement(file_path)
    return bank_statement[bank_statement.SOURCE == loan_name]


def add_config_upload_download():
    """Add config download and upload functionality to sidebar."""
    st.sidebar.markdown("### Configuration Management")

    # Download current config
    if st.sidebar.button("Download Config File"):
        try:
            # Read the config.py file directly
            with open('config.py', 'r') as file:
                config_content = file.read()

            # Create download button
            st.sidebar.download_button(
                label="Save Config File",
                data=config_content,
                file_name="config.py",
                mime="text/plain",
                key="download_config"
            )
        except Exception as e:
            st.sidebar.error(f"Error reading config file: {e}")

    # Upload config
    uploaded_file = st.sidebar.file_uploader("Upload Config File", type=["py"])
    if uploaded_file is not None:
        try:
            # Read uploaded content
            config_content = uploaded_file.read().decode('utf-8')

            # Create a namespace to execute the code
            namespace = {
                'datetime': datetime,
                'pd': pd,
                'relativedelta': relativedelta
            }

            # Execute the code in the namespace
            exec(config_content, namespace)

            # Extract the DEFAULT_CONFIG from the namespace
            if 'DEFAULT_CONFIG' in namespace:
                uploaded_config = namespace['DEFAULT_CONFIG']
                st.sidebar.success("Config loaded successfully!")
                return uploaded_config
            else:
                st.sidebar.error("Uploaded file doesn't contain DEFAULT_CONFIG.")

        except Exception as e:
            st.sidebar.error(f"Error loading config: {e}")

    # Return None if no upload or if there was an error
    return None


def main():
    file_path = 'SAV Fund & Projects - Reconciliation - 2025.03.xlsx'
    bank_statement_df = load_bank_statement(file_path)
    senior_loan_statement_df = load_loan_statement(file_path, loan_name='OakNorth Loan Account')
    mezzanine_loan_statement_df = load_loan_statement(file_path, loan_name='Coutts Loan Account')
    # Add config upload/download functionality and get updated config
    # updated_config = add_config_upload_download(DEFAULT_CONFIG)
    uploaded_config = add_config_upload_download()
    config_to_use = uploaded_config if uploaded_config is not None else DEFAULT_CONFIG

    # add_config_download()
    # user_inputs = create_input_form(DEFAULT_CONFIG)
    user_inputs = create_input_form(config_to_use)
    # Create integrated CashflowGenerator instance
    cashflow_gen = CashflowGenerator(
        bank_statement_df=bank_statement_df,
        senior_loan_statement_df=senior_loan_statement_df,
        mezzanine_loan_statement_df=mezzanine_loan_statement_df,
        **user_inputs
    )

    # Generate both cashflows
    cashflow_df = cashflow_gen.generate_cashflow()

    # Apply development cost adjustment
    if user_inputs['development_cost_adjustment'] != 0:
        adjustment = user_inputs['development_cost_adjustment']

        cashflow_df.loc['Development', 'Total'] -= adjustment

        # Adjust Total Project Cashflow similarly
        if 'Total Project Cashflow' in cashflow_df.index:
            cashflow_df.loc['Total Project Cashflow', 'Total'] -= adjustment
        cashflow_gen.project_cashflow_df = cashflow_df
    debt_cashflow_df = cashflow_gen.generate_debt_cashflow()
    equity_cashflow_df = cashflow_gen.generate_equity_cashflow()
    equity_split_data = cashflow_gen.equity_split_data if hasattr(cashflow_gen, 'equity_split_data') else {}

    # Pass the generator to the dashboard
    create_dashboard(
        cashflow_df,
        debt_cashflow_df=debt_cashflow_df,
        equity_cashflow_df=equity_cashflow_df,
        equity_split_data=equity_split_data,
        cashflow_generator=cashflow_gen,  # Pass the generator instead of loan statements
        annual_base_rate=user_inputs['annual_base_rate'],
        mc_config=DEFAULT_CONFIG['monte_carlo'],
    )


if __name__ == "__main__":
    import streamlit_analytics2 as streamlit_analytics

    with streamlit_analytics.track():
        main()
