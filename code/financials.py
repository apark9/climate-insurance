import pandas as pd
import logging
import os

# Ensure directories exist
os.makedirs("output", exist_ok=True)

def load_share_price_data(file_path):
    """
    Load and process share price data.
    """
    try:
        logging.info("Loading share price data.")
        share_price_data = pd.read_csv(file_path)

        # Convert 'Pricing Date' to datetime
        share_price_data['Pricing Date'] = pd.to_datetime(share_price_data['Pricing Date'])

        # Add a 'Year' column for grouping
        share_price_data['Year'] = share_price_data['Pricing Date'].dt.year

        # Clean share price data: convert columns to numeric and handle non-numeric values
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

        # Take the average of daily share prices for each year
        aggregated_share_prices = share_price_data.groupby('Year').agg({
            'PGR-Share Price (Daily)(%)': 'mean',
            'AFG-Share Price (Daily)(%)': 'mean',
            'ALL-Share Price (Daily)(%)': 'mean',
            'AFL-Share Price (Daily)(%)': 'mean',
            'TRV-Share Price (Daily)(%)': 'mean',
            'S&P United States BMI Insurance (Industry Group) Index-Index Value (Daily)(%)': 'mean'  # Benchmark
        }).reset_index()

        # Calculate the average share price across all tickers
        aggregated_share_prices['Average_Share_Price'] = aggregated_share_prices[
            ['PGR-Share Price (Daily)(%)', 'AFG-Share Price (Daily)(%)',
             'ALL-Share Price (Daily)(%)', 'AFL-Share Price (Daily)(%)',
             'TRV-Share Price (Daily)(%)']
        ].mean(axis=1)

        # Save the processed data
        output_path = "output/aggregated_share_price_data.csv"
        aggregated_share_prices.to_csv(output_path, index=False)
        logging.info(f"Share price data successfully processed and saved to {output_path}.")
        return aggregated_share_prices
    except Exception as e:
        logging.error(f"Error processing share price data: {e}")
        raise

def run_financial_analysis():
    """
    Run the financial analysis pipeline.
    """
    try:
        logging.info("Starting financial analysis pipeline.")

        # File path
        share_price_file_path = "data/stock_prices.csv"

        # Process share price data
        load_share_price_data(share_price_file_path)

        logging.info("Financial analysis pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Error in financial analysis pipeline: {e}")
        raise

if __name__ == "__main__":
    run_financial_analysis()
