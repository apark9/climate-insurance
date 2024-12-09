import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
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

def preprocess_sentiment_data(sentiment_data):
    """
    Preprocess sentiment data by averaging sentiment across all tickers for each year.
    """
    try:
        # Group by year and calculate the average compound sentiment
        averaged_sentiment_data = (
            sentiment_data.groupby("Year", as_index=False)["compound_sentiment"]
            .mean()
        )
        logging.info("Averaged sentiment data across all tickers by year.")
        return averaged_sentiment_data
    except Exception as e:
        logging.error(f"Error processing sentiment data: {e}")
        raise

def plot_sentiment_and_stock_prices(sentiment_data, financial_data, output_folder="plots"):
    """
    Plot sentiment trends (averaged across all tickers) alongside stock prices with dual y-axes.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Preprocess sentiment data to average sentiment across tickers
    averaged_sentiment_data = preprocess_sentiment_data(sentiment_data)

    # Merge data
    merged_data = pd.merge(averaged_sentiment_data, financial_data, on="Year", how="inner")

    # Create a plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot sentiment trends on the left y-axis
    ax1.plot(
        merged_data["Year"],
        merged_data["compound_sentiment"],
        marker="o",
        color="blue",
        label="Average Sentiment Score"
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
    if "S&P United States BMI Insurance (Industry Group) Index-Index Value (Daily)(%)" in merged_data:
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

def plot_sentiment_and_disasters(sentiment_data, disaster_data, output_folder="plots"):
    """
    Plot sentiment trends (averaged across all tickers) alongside climate disasters with dual y-axes.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Preprocess sentiment data to average sentiment across tickers
    averaged_sentiment_data = preprocess_sentiment_data(sentiment_data)

    # Merge data
    merged_data = pd.merge(averaged_sentiment_data, disaster_data, on="Year", how="inner")

    # Create a plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot sentiment trends on the left y-axis
    ax1.plot(
        merged_data["Year"],
        merged_data["compound_sentiment"],
        marker="o",
        color="blue",
        label="Average Sentiment Score"
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

def final_graph_corr(sentiment_data, disaster_data, financial_data, output_folder="plots"):
    """
    Calculate and plot correlations between sentiment score, total damages, and stock prices.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Preprocess sentiment data to average sentiment across tickers
    averaged_sentiment_data = preprocess_sentiment_data(sentiment_data)

    # Merge all data
    merged_data = pd.merge(averaged_sentiment_data, disaster_data, on="Year", how="inner")
    merged_data = pd.merge(merged_data, financial_data, on="Year", how="inner")

    # Calculate correlation matrix
    correlation_matrix = merged_data[[
        "compound_sentiment",
        "Total_Damage_Adjusted",
        "Average_Share_Price"
    ]].corr()

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix: Sentiment, Total Damages, and Stock Prices")
    plt.tight_layout()

    # Save the heatmap
    heatmap_path = os.path.join(output_folder, f"correlation_heatmap_{keyword_flag}.png")
    try:
        plt.savefig(heatmap_path, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved correlation heatmap at {heatmap_path}.")
    except Exception as e:
        logging.error(f"Error saving correlation heatmap: {e}")

    # Plot sentiment, damages, and stock prices with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot sentiment trends on the left y-axis
    ax1.plot(
        merged_data["Year"],
        merged_data["compound_sentiment"],
        marker="o",
        color="blue",
        label="Average Sentiment Score"
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Sentiment Score", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid()

    # Plot damages and stock prices on the right y-axis
    ax2 = ax1.twinx()
    ax2.plot(
        merged_data["Year"],
        merged_data["Total_Damage_Adjusted"] / 1e6,
        marker="x",
        linestyle="--",
        color="red",
        label="Total Damage (Millions)"
    )
    ax2.plot(
        merged_data["Year"],
        merged_data["Average_Share_Price"],
        marker="o",
        color="green",
        label="Average Share Price"
    )
    ax2.set_ylabel("Damages (Millions USD) / Stock Prices", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Sentiment, Total Damages, and Stock Prices")
    fig.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_folder, f"final_graph_corr_{keyword_flag}.png")
    try:
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved sentiment, damages, and stock prices plot at {plot_path}.")
    except Exception as e:
        logging.error(f"Error saving sentiment, damages, and stock prices plot: {e}")

def perform_graphing():
    """
    Perform all graphing tasks including sentiment, stock prices, and climate disasters.
    """
    # File paths
    sentiment_file = f"output/aggregated_sentiment_data_{keyword_flag}.csv"
    disaster_file = f"output/aggregated_disaster_data_{keyword_flag}.csv"
    financial_file = f"output/aggregated_share_price_data_{keyword_flag}.csv"

    # Load data
    sentiment_data = load_data(sentiment_file)
    disaster_data = load_data(disaster_file)
    financial_data = load_data(financial_file)

    try:
        plot_sentiment_and_stock_prices(sentiment_data, financial_data)
        plot_sentiment_and_disasters(sentiment_data, disaster_data)
        final_graph_corr(sentiment_data, disaster_data, financial_data)
        logging.info("Graphing completed successfully.")
    except Exception as e:
        logging.error(f"Error in graphing: {e}")

if __name__ == "__main__":
    perform_graphing()
