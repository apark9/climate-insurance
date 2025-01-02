import pandas as pd
from datetime import datetime
import logging
import os

os.makedirs("output", exist_ok=True)

key_line_items_common = [
    "Total Loss and LAE, mm",
    "Reserve Ratio, %",
    "Solvency Ratio, %",
    "Effective Interest Rate on Debt, %",
    "Deferred Acquisition Costs",
    "Payables for reinsurance premiums",
    "Total Deferred Acquisition Costs, mm",
    "Total Gross Written Premiums, mm",
    "Total Net Written Premiums, mm",
    "Net Earned Premiums",
    "Net Investment Income",
    "Net Investment Gains",
    "Book Value per Common Share",
    "Consensus Estimates - Book Value per Common Share",
    "Return on Average Common Equity, %",
    "Consensus Estimates - Return on Average Common Equity, %",
    "Consensus Estimates - Net Revenue"
]

key_line_items_afg = key_line_items_common + [
    "Property and transportation loss and LAE catastrophe losses, mm",
    "Specialty casualty loss and LAE catastrophe losses, mm",
    "Specialty financial loss and LAE catastrophe losses, mm",
    "Property and transportation combined ratio, %",
    "Specialty casualty combined ratio, %"
]

key_line_items_trv = key_line_items_common + [
    "Business Insurance - Catastrophes, net of reinsurance, mm",
    "Bond & Specialty Insurance - Catastrophes, net of reinsurance, mm",
    "Personal Insurance - Catastrophes, net of reinsurance, mm",
    "Catastrophes, net of reinsurance, %",
    "Reinsurance premium costs, mm"
]

key_line_items_afl = key_line_items_common + [
    "Aflac Japan - Benefits and claims, mm",
    "Aflac U.S - Benefits and claims, mm",
    "Corporate and Other - Benefits and claims, mm",
    "Debt to Capital Ratio, %"
]

key_line_items_pgr = key_line_items_common + [
    "Y/Y Total Loss and LAE growth, %",
    "Total Underwriting Expense, mm",
    "Underwriting Margin, %"
]

key_line_items_all = key_line_items_common + [
    "Allstate Protection - Auto - Effect of catastrophe losses, mm",
    "Gross Loss Reserves, mm",
    "Loss Payout Ratio, %"
]

company_key_line_items = {
    "AFG": key_line_items_afg,
    "TRV": key_line_items_trv,
    "AFL": key_line_items_afl,
    "PGR": key_line_items_pgr,
    "ALL": key_line_items_all,
}

def format_row_4_as_dates(header):
    try:
        row_4 = header.iloc[3, :].astype(str)

        def parse_date(value):
            try:
                return datetime.strptime(value.split()[0], '%Y-%m-%d').strftime('%Y-%m-%d')
            except ValueError:
                return value

        formatted_dates = row_4.apply(parse_date)
        header.iloc[3, :] = formatted_dates
    except Exception as e:
        print(f"ERROR - Error formatting row 4: {e}")
    return header

def filter_and_export(file_path, key_items, output_path):
    try:
        df = pd.read_excel(file_path, sheet_name='Model', header=None, engine='openpyxl')
        header = df.iloc[:4, :].copy()
        header = format_row_4_as_dates(header)
        data = df.iloc[4:, :].copy()
        filtered_data = data[data.iloc[:, 0].isin(key_items)]
        filtered_df = pd.concat([header, filtered_data], ignore_index=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        filtered_df.to_excel(output_path, index=False, header=False, engine='openpyxl')
    except Exception as e:
        print(f"ERROR - Error in financial analysis: {e}")

def compiled(output_folder, compiled_file_path):
    try:
        excel_files = [file for file in os.listdir(output_folder) if file.endswith('Filtered.xlsx')]
        compiled_data = []
        for file in excel_files:
            file_path = os.path.join(output_folder, file)
            df = pd.read_excel(file_path, sheet_name='Sheet1', header=None, engine='openpyxl')
            company_name = file.replace('.xlsx', '').replace(' Filtered', '')
            df.insert(0, 'Company', company_name)
            compiled_data.append(df)
        combined_df = pd.concat(compiled_data, ignore_index=True)
        combined_df = combined_df.dropna(how='all', subset=combined_df.columns[2:])
        combined_df.to_excel(compiled_file_path, index=False, header=False, engine='openpyxl')
        print(f"Compiled companies file successfully created at: {compiled_file_path}")
    except Exception as e:
        print(f"ERROR - Error compiling climate risks data: {e}")

def load_share_price_data(file_path):
    try:
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
        output_path = "output/aggregated_share_price_data.csv"
        aggregated_share_prices.to_csv(output_path, index=False)
        logging.info(f"Share price data saved to {output_path}.")
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
    file_paths = {
        "AFG": "data/models/American Financial Group AFG US.xlsx",
        "TRV": "data/models/The Travelers Companies TRV US.xlsx",
        "AFL": "data/models/Aflac AFL US.xlsx",
        "PGR": "data/models/The Progressive Corporation PGR US.xlsx",
        "ALL": "data/models/The Allstate Corporation ALL US.xlsx"
    }
    for company, file_path in file_paths.items():
        output_path = f"data/Financial_Models/{company} Filtered.xlsx"
        filter_and_export(file_path, company_key_line_items[company], output_path)
    compiled_file_path = "output/compiled_companies.xlsx"
    compiled("data/Financial_Models", compiled_file_path)

if __name__ == "__main__":
    run_financial_analysis()
    run_models()
