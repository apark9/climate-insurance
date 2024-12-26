import pandas as pd
import logging
import os
from config import keyword_flag

os.makedirs("output", exist_ok=True)

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

if __name__ == "__main__":
    run_financial_analysis()