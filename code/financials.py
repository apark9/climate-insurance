import pandas as pd
from datetime import datetime
import logging
import os
from config import keyword_flag

os.makedirs("output", exist_ok=True)

# Common Key Line Items
key_line_items_common = [
    # Climate Risk Indicators
    "Total Loss and LAE, mm",
    "Reserve Ratio, %",
    "Solvency Ratio, %",

    # Mitigation Efforts
    "Effective Interest Rate on Debt, %",
    "Deferred Acquisition Costs",
    "Payables for reinsurance premiums",
    "Total Deferred Acquisition Costs, mm",

    # Geographic Exposure
    "Total Gross Written Premiums, mm",
    "Total Net Written Premiums, mm",
    "Net Earned Premiums",
    "Net Investment Income",
    "Net Investment Gains",

    # Additional Key Metrics
    "Book Value per Common Share",
    "Consensus Estimates - Book Value per Common Share",
    "Return on Average Common Equity, %",
    "Consensus Estimates - Return on Average Common Equity, %",
    "Consensus Estimates - Net Revenue"
]

# Updated Key Line Items for Each Company Including Common Items
key_line_items_afg = key_line_items_common + [
    # Specific to AFG
    "Property and transportation loss and LAE catastrophe losses, mm",
    "Specialty casualty loss and LAE catastrophe losses, mm",
    "Specialty financial loss and LAE catastrophe losses, mm",
    "Property and transportation combined ratio, %",
    "Specialty casualty combined ratio, %"
]

key_line_items_trv = key_line_items_common + [
    # Specific to TRV
    "Business Insurance - Catastrophes, net of reinsurance, mm",
    "Bond & Specialty Insurance - Catastrophes, net of reinsurance, mm",
    "Personal Insurance - Catastrophes, net of reinsurance, mm",
    "Catastrophes, net of reinsurance, %"
]

key_line_items_afl = key_line_items_common + [
    # Specific to AFL
    "Aflac Japan - Benefits and claims, mm",
    "Aflac U.S - Benefits and claims, mm",
    "Corporate and Other - Benefits and claims, mm",
    "Debt to Capital Ratio, %"
]

key_line_items_pgr = key_line_items_common + [
    # Specific to PGR
    "Y/Y Total Loss and LAE growth, %",
    "Total Underwriting Expense, mm",
    "Underwriting Margin, %"
]

key_line_items_all = key_line_items_common + [
    # Specific to ALL
    "Allstate Protection - Auto - Effect of catastrophe losses, mm",
    "Gross Loss Reserves, mm",
    "Loss Payout Ratio, %"
]

def format_row_4_as_dates(header):
    try:
        row_4 = header.iloc[3, :].astype(str)

        def parse_date(value):
            try:
                return datetime.strptime(value.split()[0], '%m/%d/%Y').strftime('%Y-%m-%d')
            except ValueError:
                return value

        formatted_dates = row_4.apply(parse_date)
        header = header.copy()
        header.iloc[3, :] = formatted_dates
    except Exception as e:
        print(f"ERROR - Error formatting row 4: {e}")
    return header

def filter_and_export(file_path, key_items, output_path):
    try:
        # Load the Excel file
        df = pd.read_excel(file_path, sheet_name='Model', header=None, engine='openpyxl')

        # Extract the header (e.g., row with dates or other details)
        header = df.iloc[:4, :].copy()  # Assume the first 4 rows contain the header
        header = format_row_4_as_dates(header)  # Format row 4 dates
        data = df.iloc[4:, :].copy()    # Remaining rows contain the data

        # Filter rows where the first column matches the key items
        filtered_data = data[data.iloc[:, 0].isin(key_items)]

        # Reattach the header
        filtered_df = pd.concat([header, filtered_data], ignore_index=True)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Export to a new Excel file
        filtered_df.to_excel(output_path, index=False, header=False, engine='openpyxl')
    except Exception as e:
        print(f"ERROR - Error in financial analysis: {e}")

def extract_headers(output_folder):
    try:
        # Get all Excel files in the output folder
        excel_files = [file for file in os.listdir(output_folder) if file.endswith('.xlsx')]

        # Dictionary to store headers from each file
        headers = {}

        # Loop through each Excel file and extract headers
        for file in excel_files:
            file_path = os.path.join(output_folder, file)
            df = pd.read_excel(file_path, sheet_name='Sheet1', header=None, engine='openpyxl')

            # Extract headers as columns (first column of the file)
            headers[file] = df.iloc[:, 0].copy()

        return headers
    except Exception as e:
        print(f"ERROR - Error extracting headers: {e}")
        return {}

def create_comparison_file(output_folder, comparison_file_path):
    try:
        # Load all filtered Excel files
        excel_files = [file for file in os.listdir(output_folder) if file.endswith('.xlsx')]

        comparison_data = []

        for file in excel_files:
            file_path = os.path.join(output_folder, file)
            df = pd.read_excel(file_path, sheet_name='Sheet1', header=None, engine='openpyxl')

            # Filter rows for common key line items
            common_data = df[df.iloc[:, 0].isin(key_line_items_common)]
            common_data.insert(0, 'Company', file.replace('.xlsx', ''))  # Add company name as the first column
            comparison_data.append(common_data)

        # Combine all data into a single DataFrame
        combined_df = pd.concat(comparison_data, ignore_index=True)

        # Export to a comparison Excel file
        combined_df.to_excel(comparison_file_path, index=False, header=False, engine='openpyxl')
        print(f"Comparison file created at: {comparison_file_path}")
    except Exception as e:
        print(f"ERROR - Error creating comparison file: {e}")

def load_share_price_data(file_path):
    try:
        logging.info("Loading share price data.")
        share_price_data = pd.read_csv(file_path)

        share_price_data['Pricing Date'] = pd.to_datetime(share_price_data['Pricing Date'])
        share_price_data['Year'] = share_price_data['Pricing Date'].dt.year

        price_columns = [
            'PGR-Share Price (Daily)(%)',
            'AFG-Share Price (Daily)(%)',
            'ALL-Share Price (Daily)(%)',
            'AFL-Share Price (Daily)(%)',
            'TRV-Share Price (Daily)(%)',
            'S&P United States BMI Insurance (Industry Group) Index-Index Value (Daily)(%)'
        ]

        for col in price_columns:
            share_price_data[col] = pd.to_numeric(share_price_data[col], errors='coerce')

        aggregated_share_prices = share_price_data.groupby('Year').agg({
            'PGR-Share Price (Daily)(%)': 'mean',
            'AFG-Share Price (Daily)(%)': 'mean',
            'ALL-Share Price (Daily)(%)': 'mean',
            'AFL-Share Price (Daily)(%)': 'mean',
            'TRV-Share Price (Daily)(%)': 'mean',
            'S&P United States BMI Insurance (Industry Group) Index-Index Value (Daily)(%)': 'mean'
        }).reset_index()

        aggregated_share_prices['Average_Share_Price'] = aggregated_share_prices[
            ['PGR-Share Price (Daily)(%)', 'AFG-Share Price (Daily)(%)',
             'ALL-Share Price (Daily)(%)', 'AFL-Share Price (Daily)(%)',
             'TRV-Share Price (Daily)(%)']
        ].mean(axis=1)

        output_path = f"output/aggregated_share_price_data_{keyword_flag}.csv"
        aggregated_share_prices.to_csv(output_path, index=False)
        logging.info(f"Share price data successfully processed and saved to {output_path}.")
        return aggregated_share_prices
    except Exception as e:
        logging.error(f"Error processing share price data: {e}")
        raise

def run_financial_analysis():
    try:
        logging.info("Starting financial analysis pipeline.")
        share_price_file_path = "data/stock_prices.csv"
        load_share_price_data(share_price_file_path)
        logging.info("Financial analysis pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Error in financial analysis pipeline: {e}")
        raise

def run_models():
    # File paths for the input Excel files with full descriptive names
    file_paths = {
        "AFG": "data/Financial_Models/American Financial Group AFG US.xlsx",
        "TRV": "data/Financial_Models/The Travelers Companies TRV US.xlsx",
        "AFL": "data/Financial_Models/Aflac AFL US.xlsx",
        "PGR": "data/Financial_Models/The Progressive Corporation PGR US.xlsx",
        "ALL": "data/Financial_Models/The Allstate Corporation ALL US.xlsx"
    }

    # Corresponding output paths in the "output" folder
    output_paths = {
        "AFG": "output/American_Financial_Group_AFG_US.xlsx",
        "TRV": "output/The_Travelers_Companies_TRV_US.xlsx",
        "AFL": "output/Aflac_AFL_US.xlsx",
        "PGR": "output/The_Progressive_Corporation_PGR_US.xlsx",
        "ALL": "output/The_Allstate_Corporation_ALL_US.xlsx"
    }

    # Filter and export individual company data
    filter_and_export(file_paths["AFG"], key_line_items_afg, output_paths["AFG"])
    filter_and_export(file_paths["TRV"], key_line_items_trv, output_paths["TRV"])
    filter_and_export(file_paths["AFL"], key_line_items_afl, output_paths["AFL"])
    filter_and_export(file_paths["PGR"], key_line_items_pgr, output_paths["PGR"])
    filter_and_export(file_paths["ALL"], key_line_items_all, output_paths["ALL"])

    # Create comparison file for common key line items
    create_comparison_file("output", "output/comparison_file.xlsx")

if __name__ == "__main__":
    run_financial_analysis()
    run_models()
