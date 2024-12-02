import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
import ast

def load_data():
    """
    Load data from the CSV file.
    """
    try:
        df = pd.read_csv("/export/home/rcsguest/rcs_apark/Desktop/home-insurance/output/insurance_transcripts.csv")
        logging.info("Loaded data from 'insurance_transcripts.csv'.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def analyze_sentiment_trends(data):
    """
    Analyze sentiment trends over time by grouping by ticker and year.
    """
    # Parse sentiment strings into dictionaries if necessary
    if isinstance(data["sentiment"].iloc[0], str):
        data["sentiment"] = data["sentiment"].apply(ast.literal_eval)

    # Ensure sentiment is a numerical value (compound score)
    data["compound_sentiment"] = data["sentiment"].apply(
        lambda x: x["compound"] if isinstance(x, dict) else None
    )

    # Drop rows where compound_sentiment is missing
    data = data.dropna(subset=["compound_sentiment"])

    # Group by ticker and year, and calculate mean sentiment
    sentiment_over_time = (
        data.groupby(["ticker", "year"])["compound_sentiment"]
        .mean()
        .reset_index()
    )
    logging.info("Calculated sentiment trends over time.")
    return sentiment_over_time

def plot_sentiment_trends(data, output_folder="plots"):
    """
    Plot sentiment trends for each ticker and save to the specified folder.
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists

    for ticker in data["ticker"].unique():
        ticker_data = data[data["ticker"] == ticker]

        if ticker_data.empty:
            logging.warning(f"No data available for ticker: {ticker}")
            continue

        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.plot(
            ticker_data["year"],
            ticker_data["compound_sentiment"],
            marker="o",
            label="Compound Sentiment"
        )
        plt.title(f"Sentiment Trend for {ticker}")
        plt.xlabel("Year")
        plt.ylabel("Sentiment Score")
        plt.grid()
        plt.legend()

        # Save the plot explicitly to the output folder
        plot_path = os.path.join(output_folder, f"{ticker}_sentiment_trend.png")
        try:
            plt.savefig(plot_path)
            plt.close()
            logging.info(f"Saved sentiment trend plot for {ticker} at {plot_path}.")
        except Exception as e:
            logging.error(f"Error saving plot for {ticker} to {plot_path}: {e}")

import matplotlib.ticker as mticker

def plot_all_tickers(data, output_folder="plots"):
    """
    Plot sentiment trends for all tickers on a single graph and save the output.
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists

    plt.figure(figsize=(10, 8))

    for ticker in data["ticker"].unique():
        ticker_data = data[data["ticker"] == ticker]

        if ticker_data.empty:
            logging.warning(f"No data available for ticker: {ticker}")
            continue

        # Plot the trend for this ticker
        plt.plot(
            ticker_data["year"],
            ticker_data["compound_sentiment"],
            marker="o",
            label=ticker
        )

    # Set x-axis ticks to whole years
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Add labels, title, and legend
    plt.title("Sentiment Trends Across All Tickers")
    plt.xlabel("Year")
    plt.ylabel("Sentiment Score")
    plt.grid()
    plt.legend(title="Ticker", loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Save the plot explicitly to the output folder
    plot_path = os.path.join(os.path.abspath(output_folder), "all_tickers_sentiment_trends.png")
    try:
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved combined sentiment trend plot for all tickers at {plot_path}.")
    except Exception as e:
        logging.error(f"Error saving combined plot to {plot_path}: {e}")


def perform_analysis():
    """
    Perform the sentiment analysis and generate plots.
    """
    data = load_data()
    sentiment_trends = analyze_sentiment_trends(data)
    plot_sentiment_trends(sentiment_trends)
    plot_all_tickers(sentiment_trends)

if __name__ == "__main__":
    perform_analysis()
