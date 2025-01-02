import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from config import keyword_flag

os.makedirs("output", exist_ok=True)
os.makedirs("plots", exist_ok=True)

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Successfully loaded data from {file_path}.")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise

def plot_sentiment_and_stock_prices(merged_data, output_folder="plots"):
    aggregated_data = merged_data.groupby("Year").agg({
        "compound_sentiment": "mean",
        "Average_Share_Price": "mean",
        "S&P United States BMI Insurance (Industry Group) Index-Index Value (Daily)(%)": "mean"
    }).reset_index()

    aggregated_data = aggregated_data.sort_values("Year")

    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax1.plot(
        aggregated_data["Year"],
        aggregated_data["compound_sentiment"],
        marker="o",
        color="blue",
        label="Average Sentiment Score (Primary)"
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Sentiment Score", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid()

    ax2 = ax1.twinx()
    ax2.plot(
        aggregated_data["Year"],
        aggregated_data["Average_Share_Price"],
        marker="o",
        color="green",
        label="Average Share Price (Secondary)"
    )
    if "S&P United States BMI Insurance (Industry Group) Index-Index Value (Daily)(%)" in aggregated_data:
        ax2.plot(
            aggregated_data["Year"],
            aggregated_data["S&P United States BMI Insurance (Industry Group) Index-Index Value (Daily)(%)"],
            linestyle="--",
            color="orange",
            label="S&P Insurance BMI (Secondary)"
        )
    ax2.set_ylabel("Stock Prices", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.xticks(aggregated_data["Year"], aggregated_data["Year"].astype(int), rotation=45)

    plt.title("Sentiment Trends and Stock Prices")
    fig.tight_layout()
    output_path = os.path.join(output_folder, f"sentiment_and_stock_prices_{keyword_flag}.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved sentiment and stock prices plot at {output_path}.")

def plot_sentiment_and_disasters(merged_data, output_folder="plots"):
    aggregated_data = merged_data.groupby("Year").agg({
        "compound_sentiment": "mean",
        "Disaster_Count": "mean",
        "Total_Damage_Adjusted": "mean"
    }).reset_index()

    aggregated_data = aggregated_data.sort_values("Year")

    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax1.plot(
        aggregated_data["Year"],
        aggregated_data["compound_sentiment"],
        marker="o",
        color="blue",
        label="Average Sentiment Score (Primary)"
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Sentiment Score", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid()

    ax2 = ax1.twinx()
    ax2.bar(
        aggregated_data["Year"],
        aggregated_data["Disaster_Count"],
        alpha=0.5,
        color="orange",
        label="Number of Disasters (Secondary)"
    )
    ax2.plot(
        aggregated_data["Year"],
        aggregated_data["Total_Damage_Adjusted"] / 1e6,
        marker="x",
        color="red",
        linestyle="--",
        label="Total Damage (Millions, Secondary)"
    )
    ax2.set_ylabel("Disasters / Damage (Millions USD)", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.xticks(aggregated_data["Year"], aggregated_data["Year"].astype(int), rotation=45)

    plt.title("Sentiment Trends and Climate Disasters")
    fig.tight_layout()
    output_path = os.path.join(output_folder, f"sentiment_vs_disasters_{keyword_flag}.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved sentiment vs disasters plot at {output_path}.")

def plot_language_complexity(merged_data, output_folder="plots"):
    try:
        logging.info(f"Data columns in plot_language_complexity: {merged_data.columns}")

        aggregated_data = merged_data.groupby("Year").agg({
            "flesch_reading_ease": "mean",
            "gunning_fog_index": "mean",
            "smog_index": "mean",
            "lexical_density": "mean"
        }).reset_index()

        fig, ax1 = plt.subplots(figsize=(12, 8))

        ax1.plot(
            aggregated_data["Year"], 
            aggregated_data["flesch_reading_ease"], 
            marker="o", 
            label="Flesch Reading Ease (Primary)", 
            color="blue"
        )
        ax1.plot(
            aggregated_data["Year"], 
            aggregated_data["gunning_fog_index"], 
            marker="o", 
            label="Gunning Fog Index (Primary)", 
            color="orange"
        )
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Primary Metrics", color="black")
        ax1.tick_params(axis="y", labelcolor="black")
        ax1.grid()

        ax2 = ax1.twinx()
        ax2.plot(
            aggregated_data["Year"], 
            aggregated_data["smog_index"], 
            marker="x", 
            linestyle="--", 
            label="SMOG Index (Secondary)", 
            color="green"
        )
        ax2.plot(
            aggregated_data["Year"], 
            aggregated_data["lexical_density"], 
            marker="s", 
            linestyle="-.", 
            label="Lexical Density (Secondary)", 
            color="red"
        )
        ax2.set_ylabel("Secondary Metrics", color="black")
        ax2.tick_params(axis="y", labelcolor="black")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)

        plt.xticks(aggregated_data["Year"], aggregated_data["Year"].astype(int), rotation=45)

        plt.title("Language Complexity Metrics Over Time")
        plt.tight_layout()

        output_path = os.path.join(output_folder, f"language_complexity_{keyword_flag}.png")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved language complexity plot to {output_path}.")
    except Exception as e:
        logging.error(f"Error plotting language complexity: {e}")

def perform_graphing():
    sentiment_file = f"output/aggregated_transcript_analysis_results_{keyword_flag}.csv"
    disaster_file = f"output/aggregated_disaster_data_{keyword_flag}.csv"
    financial_file = f"output/aggregated_share_price_data_{keyword_flag}.csv"

    sentiment_data = load_data(sentiment_file)
    disaster_data = load_data(disaster_file)
    financial_data = load_data(financial_file)

    merged_data = pd.merge(sentiment_data, disaster_data, on="Year", how="inner")
    merged_data = pd.merge(merged_data, financial_data, on="Year", how="inner")

    try:
        plot_sentiment_and_stock_prices(merged_data)
        plot_sentiment_and_disasters(merged_data)
        plot_language_complexity(sentiment_data)
        logging.info("Graphing completed successfully.")
    except Exception as e:
        logging.error(f"Error in graphing: {e}")

if __name__ == "__main__":
    perform_graphing()
