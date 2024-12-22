import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from config import keyword_flag

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

def plot_sentiment_and_stock_prices(merged_data, output_folder="plots"):
    fig, ax1 = plt.subplots(figsize=(12, 8))

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

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Sentiment Trends and Stock Prices")
    fig.tight_layout()

    output_path = os.path.join(output_folder, f"sentiment_and_stock_prices_{keyword_flag}.png")
    try:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved sentiment and stock prices plot at {output_path}.")
    except Exception as e:
        logging.error(f"Error saving sentiment and stock prices plot: {e}")

def plot_sentiment_and_disasters(merged_data, output_folder="plots"):
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
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

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Sentiment Trends and Climate Disasters")
    fig.tight_layout()

    output_path = os.path.join(output_folder, f"sentiment_vs_disasters_{keyword_flag}.png")
    try:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved sentiment vs disasters plot at {output_path}.")
    except Exception as e:
        logging.error(f"Error saving sentiment vs disasters plot: {e}")

def final_graph_corr(merged_data, output_folder="plots"):
    """
    Calculate and plot correlations between sentiment score, total damages, and stock prices.
    """
    os.makedirs(output_folder, exist_ok=True)
    correlation_matrix = merged_data[[
        "compound_sentiment",
        "Total_Damage_Adjusted",
        "Average_Share_Price"
    ]].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix: Sentiment, Total Damages, and Stock Prices")
    plt.tight_layout()

    heatmap_path = os.path.join(output_folder, f"correlation_heatmap_{keyword_flag}.png")
    try:
        plt.savefig(heatmap_path, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved correlation heatmap at {heatmap_path}.")
    except Exception as e:
        logging.error(f"Error saving correlation heatmap: {e}")

    fig, ax1 = plt.subplots(figsize=(12, 8))

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

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Sentiment, Total Damages, and Stock Prices")
    fig.tight_layout()

    plot_path = os.path.join(output_folder, f"final_graph_corr_{keyword_flag}.png")
    try:
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved sentiment, damages, and stock prices plot at {plot_path}.")
    except Exception as e:
        logging.error(f"Error saving sentiment, damages, and stock prices plot: {e}")

def plot_language_complexity(merged_data, output_folder="plots"):
    try:
        logging.info(f"Data columns in plot_language_complexity: {merged_data.columns}")

        # Aggregate metrics by year
        aggregated_data = merged_data.groupby("Year").agg({
            "flesch_reading_ease": "mean",
            "gunning_fog_index": "mean",
            "smog_index": "mean",
            "lexical_density": "mean"
        }).reset_index()

        # Set up the plot
        plt.figure(figsize=(12, 8))
        metrics = ["flesch_reading_ease", "gunning_fog_index", "smog_index", "lexical_density"]
        
        for metric in metrics:
            plt.plot(
                aggregated_data["Year"], 
                aggregated_data[metric], 
                marker="o", 
                label=metric.replace("_", " ").title()
            )
        
        # Customize the plot
        plt.title("Language Complexity Metrics Over Time")
        plt.xlabel("Year")
        plt.ylabel("Average Metric Value")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        # Save the plot
        output_path = os.path.join(output_folder, f"language_complexity_{keyword_flag}.png")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved language complexity plot to {output_path}.")
    except Exception as e:
        logging.error(f"Error plotting language complexity: {e}")

def perform_graphing():
    """
    Perform all graphing tasks including sentiment, stock prices, and climate disasters.
    """
    sentiment_file = f"output/aggregated_sentiment_data_{keyword_flag}.csv"
    disaster_file = f"output/aggregated_disaster_data_{keyword_flag}.csv"
    financial_file = f"output/aggregated_share_price_data_{keyword_flag}.csv"

    sentiment_data = load_data(sentiment_file)
    disaster_data = load_data(disaster_file)
    financial_data = load_data(financial_file)

    # Merge sentiment, disaster, and financial data
    merged_data = pd.merge(sentiment_data, disaster_data, on="Year", how="inner")
    merged_data = pd.merge(merged_data, financial_data, on="Year", how="inner")

    try:
        # Plot using merged data
        plot_sentiment_and_stock_prices(merged_data)
        plot_sentiment_and_disasters(merged_data)
        final_graph_corr(merged_data)

        # Pass sentiment_data directly for language complexity plotting
        plot_language_complexity(sentiment_data)
        
        logging.info("Graphing completed successfully.")
    except Exception as e:
        logging.error(f"Error in graphing: {e}")

if __name__ == "__main__":
    perform_graphing()
