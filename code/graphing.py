import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import ast
from config import keywords, keyword_flag

# Ensure output and plots directories exist
os.makedirs("output", exist_ok=True)
os.makedirs("plots", exist_ok=True)

def load_data(file_path):
    """
    Load data from the CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Successfully loaded data from {file_path}.")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise

def plot_sentiment_and_stock_prices(output_folder="plots"):
    """
    Plot sentiment trends alongside stock prices with dual y-axes.
    """
    os.makedirs(output_folder, exist_ok=True)

    # File paths
    sentiment_file = "output/aggregated_sentiment_data.csv"
    financial_file = "output/aggregated_share_price_data.csv"

    # Load data
    sentiment_data = load_data(sentiment_file)
    financial_data = load_data(financial_file)

    # Merge data
    merged_data = pd.merge(sentiment_data, financial_data, on="Year", how="inner")

    # Create a plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot sentiment trends on the left y-axis
    ax1.plot(
        merged_data["Year"],
        merged_data["compound_sentiment"],
        marker="o",
        color="blue",
        label="Total Sentiment (Summed)"
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Sentiment Score", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid()

    # Plot stock prices on the right y-axis
    ax2 = ax1.twinx()
    ax2.plot(
        merged_data["Year"],
        merged_data["Average_Share_Price"],
        marker="o",
        color="green",
        label="Average Share Price"
    )
    ax2.plot(
        merged_data["Year"],
        merged_data["S&P United States BMI Insurance (Industry Group) Index-Index Value (Daily)(%)"],
        linestyle="--",
        color="orange",
        label="S&P Insurance BMI (Benchmark)"
    )
    ax2.set_ylabel("Stock Prices", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Sentiment Trends and Stock Prices")
    fig.tight_layout()

    # Save the plot
    output_path = os.path.join(output_folder, f"sentiment_and_stock_prices_{keyword_flag}.png")
    try:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved sentiment and stock prices plot at {output_path}.")
    except Exception as e:
        logging.error(f"Error saving sentiment and stock prices plot: {e}")

def plot_sentiment_and_disasters(output_folder="plots"):
    """
    Plot sentiment trends alongside climate disasters with dual y-axes.
    """
    os.makedirs(output_folder, exist_ok=True)

    # File paths
    sentiment_file = "output/aggregated_sentiment_data.csv"
    disaster_file = "output/aggregated_disaster_data.csv"

    # Load data
    sentiment_data = load_data(sentiment_file)
    disaster_data = load_data(disaster_file)

    # Merge data
    merged_data = pd.merge(sentiment_data, disaster_data, on="Year", how="inner")

    # Create a plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot sentiment trends on the left y-axis
    ax1.plot(
        merged_data["Year"],
        merged_data["compound_sentiment"],
        marker="o",
        color="blue",
        label="Sentiment Score"
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Sentiment Score", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid()

    # Plot disaster counts and total damage on the right y-axis
    ax2 = ax1.twinx()
    ax2.bar(
        merged_data["Year"],
        merged_data["Disaster_Count"],
        alpha=0.5,
        color="orange",
        label="Number of Disasters"
    )
    ax2.plot(
        merged_data["Year"],
        merged_data["Total_Damage_Adjusted"] / 1e6,
        marker="x",
        color="red",
        linestyle="--",
        label="Total Damage (Millions)"
    )
    ax2.set_ylabel("Disasters / Damage (Millions USD)", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Sentiment Trends and Climate Disasters")
    fig.tight_layout()

    # Save the plot
    output_path = os.path.join(output_folder, f"sentiment_vs_disasters_{keyword_flag}.png")
    try:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved sentiment vs disasters plot at {output_path}.")
    except Exception as e:
        logging.error(f"Error saving sentiment vs disasters plot: {e}")

# def plot_sentiment_by_ticker(output_folder="plots"):
    """
    Plot sentiment trends for each ticker in separate subplots.
    """
    os.makedirs(output_folder, exist_ok=True)

    # File path
    sentiment_file = "output/aggregated_sentiment_data.csv"

    # Load data
    sentiment_data = load_data(sentiment_file)

    # Get unique tickers
    tickers = sentiment_data["ticker"].unique()

    # Create subplots for each ticker
    fig, axes = plt.subplots(len(tickers), 1, figsize=(12, 8 * len(tickers)), sharex=True)

    for i, ticker in enumerate(tickers):
        ticker_data = sentiment_data[sentiment_data["ticker"] == ticker]

        # Plot sentiment trends for each ticker
        axes[i].plot(
            ticker_data["Year"],
            ticker_data["compound_sentiment"],
            marker="o",
            label=f"{ticker} Sentiment",
            color="blue"
        )
        axes[i].set_title(f"Sentiment Trend for {ticker}")
        axes[i].set_ylabel("Sentiment Score")
        axes[i].axhline(y=0, color="black", linestyle="--", linewidth=1.5)
        axes[i].grid()
        axes[i].legend()

    # Set common x-axis label
    axes[-1].set_xlabel("Year")

    # Adjust layout and save the plot
    fig.tight_layout()
    output_path = os.path.join(output_folder, f"sentiment_by_ticker_{keyword_flag}.png")
    try:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved sentiment trends by ticker plot at {output_path}.")
    except Exception as e:
        logging.error(f"Error saving sentiment trends by ticker plot: {e}")

def perform_graphing():
    """
    Perform all graphing tasks including sentiment, stock prices, and climate disasters.
    """
    try:
        plot_sentiment_and_stock_prices()
        plot_sentiment_and_disasters()
        # plot_sentiment_by_ticker()
        logging.info("Graphing completed successfully.")
    except Exception as e:
        logging.error(f"Error in graphing: {e}")

if __name__ == "__main__":
    perform_graphing()
