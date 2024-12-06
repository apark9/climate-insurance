import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
import ast
from config import keywords

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
    Plot sentiment trends for each ticker, and save to the specified folder.
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists
    # Individual plots for each ticker
    for ticker in data["ticker"].unique():
        ticker_data = data[data["ticker"] == ticker].sort_values(by="year")

        if ticker_data.empty:
            logging.warning(f"No data available for ticker: {ticker}")
            continue

        # Plot sentiment trends
        plt.figure(figsize=(10, 6))
        plt.plot(
            ticker_data["year"],
            ticker_data["compound_sentiment"],
            marker="o",
            label="Compound Sentiment"
        )
        plt.axhline(y=0, color="black", linestyle="--", linewidth=2)  # Add thicker horizontal line at 0
        plt.title(f"Sentiment Trend for {ticker}")
        plt.xlabel("Year")
        plt.ylabel("Sentiment Score")
        plt.grid()
        plt.legend()

        # Format the x-axis to show only full years
        plt.xticks(ticker_data["year"].unique().astype(int))

        # Add keywords below the plot
        keyword_text = f"Keywords: {', '.join(keywords)}"
        plt.figtext(0.5, -0.1, keyword_text, wrap=True, horizontalalignment="center", fontsize=10)

        # Save the sentiment trend plot
        sentiment_plot_path = os.path.join(output_folder, f"{ticker}_sentiment.png")
        try:
            plt.savefig(sentiment_plot_path, bbox_inches="tight")
            plt.close()
            logging.info(f"Saved sentiment trend plot for {ticker} at {sentiment_plot_path}.")
        except Exception as e:
            logging.error(f"Error saving sentiment plot for {ticker}: {e}")

    # Combined plot for all tickers (sentiment)
    plt.figure(figsize=(10, 6))
    for ticker in data["ticker"].unique():
        ticker_data = data[data["ticker"] == ticker].sort_values(by="year")
        if ticker_data.empty:
            continue
        plt.plot(
            ticker_data["year"],
            ticker_data["compound_sentiment"],
            marker="o",
            label=ticker
        )
    plt.axhline(y=0, color="black", linestyle="--", linewidth=2)  # Add thicker horizontal line at 0
    plt.title("Sentiment Trends Across All Tickers")
    plt.xlabel("Year")
    plt.ylabel("Sentiment Score")
    plt.grid()
    plt.legend(title="Ticker")

    # Format the x-axis for full years
    plt.xticks(data["year"].unique().astype(int))
    # Add keywords below the combined plot
    keyword_text = f"Keywords: {', '.join(keywords)}"
    plt.figtext(0.5, -0.1, keyword_text, wrap=True, horizontalalignment="center", fontsize=10)

    combined_sentiment_plot_path = os.path.join(output_folder, "all_tickers_sentiment.png")
    try:
        plt.savefig(combined_sentiment_plot_path, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved combined sentiment trend plot at {combined_sentiment_plot_path}.")
    except Exception as e:
        logging.error(f"Error saving combined sentiment plot: {e}")

def plot_sentiment_trends_derivatives(data, output_folder="plots"):
    """
    Plot sentiment trends and their first derivative for each ticker, and save to the specified folder.
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure the folder 

    # Individual plots for each ticker
    for ticker in data["ticker"].unique():
        ticker_data = data[data["ticker"] == ticker].sort_values(by="year")

        if ticker_data.empty:
            logging.warning(f"No data available for ticker: {ticker}")
            continue

        # Calculate first derivative (rate of change)
        ticker_data["sentiment_derivative"] = ticker_data["compound_sentiment"].diff()

        # Plot sentiment trends
        plt.figure(figsize=(10, 6))
        plt.plot(
            ticker_data["year"],
            ticker_data["compound_sentiment"],
            marker="o",
            label="Compound Sentiment"
        )
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1.5)  # Add thicker horizontal line at 0
        plt.title(f"Sentiment Trend for {ticker}")
        plt.xlabel("Year")
        plt.ylabel("Sentiment Score")
        plt.grid()
        plt.legend()

        # Format the x-axis to show only full years
        plt.xticks(ticker_data["year"].unique().astype(int))

        # Add keywords below the plot
        keyword_text = f"Keywords: {', '.join(keywords)}"
        plt.figtext(0.5, -0.1, keyword_text, wrap=True, horizontalalignment="center", fontsize=10)

        # Save the sentiment trend plot
        sentiment_plot_path = os.path.join(output_folder, f"{ticker}_sentiment.png")
        try:
            plt.savefig(sentiment_plot_path, bbox_inches="tight")
            plt.close()
            logging.info(f"Saved sentiment trend plot for {ticker} at {sentiment_plot_path}.")
        except Exception as e:
            logging.error(f"Error saving sentiment plot for {ticker}: {e}")

        # Plot the first derivative of sentiment
        plt.figure(figsize=(10, 6))
        plt.plot(
            ticker_data["year"],
            ticker_data["sentiment_derivative"],
            marker="o",
            label="Sentiment Derivative",
            color="orange"
        )
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1.5)  # Add thicker horizontal line at 0
        plt.title(f"Rate of Change in Sentiment for {ticker}")
        plt.xlabel("Year")
        plt.ylabel("Change in Sentiment")
        plt.grid()
        plt.legend()

        # Format the x-axis for full years
        plt.xticks(ticker_data["year"].unique().astype(int))

        # Save the derivative plot
        derivative_plot_path = os.path.join(output_folder, f"{ticker}_sentiment_derivative.png")
        try:
            plt.savefig(derivative_plot_path, bbox_inches="tight")
            plt.close()
            logging.info(f"Saved sentiment derivative plot for {ticker} at {derivative_plot_path}.")
        except Exception as e:
            logging.error(f"Error saving derivative plot for {ticker}: {e}")

    # Combined plot for all tickers (sentiment derivative)
    plt.figure(figsize=(10, 6))
    for ticker in data["ticker"].unique():
        ticker_data = data[data["ticker"] == ticker].sort_values(by="year")
        if ticker_data.empty:
            continue
        ticker_data["sentiment_derivative"] = ticker_data["compound_sentiment"].diff()
        plt.plot(
            ticker_data["year"],
            ticker_data["sentiment_derivative"],
            marker="o",
            label=ticker
        )
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1.5)  # Add thicker horizontal line at 0
    plt.title("Rate of Change in Sentiment Across All Tickers")
    plt.xlabel("Year")
    plt.ylabel("Change in Sentiment")
    plt.grid()
    plt.legend(title="Ticker")

    # Format the x-axis for full years
    plt.xticks(data["year"].unique().astype(int))

    # Add keywords below the combined plot
    plt.figtext(0.5, -0.1, keyword_text, wrap=True, horizontalalignment="center", fontsize=10)

    combined_derivative_plot_path = os.path.join(output_folder, "all_tickers_sentiment_derivative.png")
    try:
        plt.savefig(combined_derivative_plot_path, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved combined sentiment derivative plot at {combined_derivative_plot_path}.")
    except Exception as e:
        logging.error(f"Error saving combined derivative plot: {e}")

def calculate_keyword_frequencies(data):
    """
    Calculate keyword frequencies for each year and ticker.
    """
    # Create a new column for keyword frequencies
    data["keyword_count"] = data["sentence"].apply(
        lambda x: sum(x.lower().count(keyword.lower()) for keyword in keywords)
    )

    # Group by ticker and year, and sum keyword counts
    keyword_frequencies = (
        data.groupby(["ticker", "year"])["keyword_count"]
        .sum()
        .reset_index()
    )
    logging.info("Calculated keyword frequencies over time.")
    return keyword_frequencies

def plot_keyword_frequencies(data, output_folder="plots"):
    """
    Plot keyword frequencies for each ticker and include the average, then save to the specified folder.
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists
    # Calculate the average keyword frequency across all tickers for each year
    average_frequencies = (
        data.groupby("year")["keyword_count"]
        .mean()
        .reset_index()
        .rename(columns={"keyword_count": "average_keyword_count"})
    )

    # Individual plots for each ticker
    for ticker in data["ticker"].unique():
        ticker_data = data[data["ticker"] == ticker].sort_values(by="year")

        if ticker_data.empty:
            logging.warning(f"No data available for ticker: {ticker}")
            continue

        # Plot keyword frequencies
        plt.figure(figsize=(10, 6))
        plt.plot(
            ticker_data["year"],
            ticker_data["keyword_count"],
            marker="o",
            label=f"{ticker} Keyword Count"
        )
        plt.axhline(y=0, color="black", linestyle="--", linewidth=2)  # Thicker horizontal line at 0
        plt.title(f"Keyword Frequency Trend for {ticker}")
        plt.xlabel("Year")
        plt.ylabel("Keyword Frequency")
        plt.grid()
        plt.legend()

        # Format the x-axis to show only full years
        plt.xticks(ticker_data["year"].unique().astype(int))

        # Add keywords below the plot
        keyword_text = f"Keywords: {', '.join(keywords)}"
        plt.figtext(0.5, -0.1, keyword_text, wrap=True, horizontalalignment="center", fontsize=10)

        # Save the keyword frequency plot
        keyword_plot_path = os.path.join(output_folder, f"{ticker}_keyword_frequency.png")
        try:
            plt.savefig(keyword_plot_path, bbox_inches="tight")
            plt.close()
            logging.info(f"Saved keyword frequency plot for {ticker} at {keyword_plot_path}.")
        except Exception as e:
            logging.error(f"Error saving keyword frequency plot for {ticker}: {e}")

    # Combined plot for all tickers with the average line
    plt.figure(figsize=(10, 6))
    for ticker in data["ticker"].unique():
        ticker_data = data[data["ticker"] == ticker].sort_values(by="year")
        if ticker_data.empty:
            continue
        plt.plot(
            ticker_data["year"],
            ticker_data["keyword_count"],
            marker="o",
            label=ticker
        )

    # Add the average line
    plt.plot(
        average_frequencies["year"],
        average_frequencies["average_keyword_count"],
        marker="o",
        linestyle="--",
        linewidth=2,
        color="purple",
        label="Average"
    )

    plt.axhline(y=0, color="black", linestyle="--", linewidth=2)  # Thicker horizontal line at 0
    plt.title("Keyword Frequency Trends Across All Tickers (With Average)")
    plt.xlabel("Year")
    plt.ylabel("Keyword Frequency")
    plt.grid()
    plt.legend(title="Ticker")

    # Format the x-axis for full years
    plt.xticks(data["year"].unique().astype(int))
    # Add keywords below the combined plot
    keyword_text = f"Keywords: {', '.join(keywords)}"
    plt.figtext(0.5, -0.1, keyword_text, wrap=True, horizontalalignment="center", fontsize=10)

    combined_keyword_plot_path = os.path.join(output_folder, "all_tickers_keyword_frequency_with_average.png")
    try:
        plt.savefig(combined_keyword_plot_path, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved combined keyword frequency plot with average at {combined_keyword_plot_path}.")
    except Exception as e:
        logging.error(f"Error saving combined keyword frequency plot with average: {e}")

def perform_graphing():
    data = load_data()
    sentiment_trends = analyze_sentiment_trends(data)
    plot_sentiment_trends(sentiment_trends)
    plot_sentiment_trends_derivatives(sentiment_trends)
    keyword_frequencies = calculate_keyword_frequencies(data)
    plot_keyword_frequencies(keyword_frequencies)

if __name__ == "__main__":
    perform_graphing()