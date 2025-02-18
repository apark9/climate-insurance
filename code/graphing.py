import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/graphing.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
print("\U0001F680 Graphing script started...", flush=True)

# Create output directories if they don't exist
os.makedirs("plots", exist_ok=True)
ANALYSIS_FOLDER = "analysis/sentiment"


def load_data(file_path):
    """Loads CSV data from a given file path."""
    try:
        data = pd.read_csv(file_path)
        logging.info(f"✅ Successfully loaded data from {file_path}.")
        return data
    except Exception as e:
        logging.error(f"❌ Error loading data from {file_path}: {e}")
        raise


def plot_sentiment_trends():
    """Plots sentiment trends over time for FinBERT and Vader."""
    sentiment_file = os.path.join(ANALYSIS_FOLDER, "sentiment_trends.csv")
    df = load_data(sentiment_file)

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="Year", y="avg_finbert", marker="o", label="FinBERT Sentiment")
    sns.lineplot(data=df, x="Year", y="avg_vader", marker="s", label="VADER Sentiment")
    
    plt.xlabel("Year")
    plt.ylabel("Average Sentiment Score")
    plt.title("Sentiment Trends Over Time (FinBERT vs. Vader)")
    plt.legend()
    plt.grid()
    
    output_path = "plots/sentiment_trends.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logging.info(f"✅ Saved sentiment trends plot at {output_path}.")


def plot_keyword_sentiment():
    """Plots average sentiment scores for different keywords."""
    keyword_file = os.path.join(ANALYSIS_FOLDER, "keyword_sentiment.csv")
    df = load_data(keyword_file)

    df_sorted = df.sort_values(by="count", ascending=False).head(20)  # Top 20 keywords

    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_sorted, y="keyword", x="avg_finbert", color="blue", label="FinBERT")
    sns.barplot(data=df_sorted, y="keyword", x="avg_vader", color="orange", label="VADER")
    
    plt.xlabel("Average Sentiment Score")
    plt.ylabel("Keyword")
    plt.title("Top Keywords by Sentiment (FinBERT vs. Vader)")
    plt.legend()
    
    output_path = "plots/keyword_sentiment.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logging.info(f"✅ Saved keyword sentiment plot at {output_path}.")


def plot_financial_vs_climate():
    """Compares sentiment for financial vs. climate-related keywords."""
    keyword_file = os.path.join(ANALYSIS_FOLDER, "keyword_sentiment.csv")
    df = load_data(keyword_file)

    financial_keywords = ["capital markets", "loss ratios", "reinsurance", "market exposure"]
    climate_keywords = ["climate change", "hurricane", "flood", "wildfire"]

    df_financial = df[df["keyword"].isin(financial_keywords)]
    df_climate = df[df["keyword"].isin(climate_keywords)]

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_financial, x="keyword", y="avg_finbert", color="blue", label="Financial (FinBERT)")
    sns.barplot(data=df_climate, x="keyword", y="avg_finbert", color="green", label="Climate (FinBERT)")
    
    plt.xlabel("Keyword")
    plt.ylabel("Average Sentiment Score")
    plt.title("Financial vs. Climate Keyword Sentiment (FinBERT)")
    plt.legend()
    
    output_path = "plots/financial_vs_climate.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logging.info(f"✅ Saved financial vs. climate sentiment plot at {output_path}.")


def plot_sentiment_distribution():
    """Plots the distribution of sentiment scores by year."""
    sentiment_file = os.path.join(ANALYSIS_FOLDER, "sentiment_trends.csv")
    df = load_data(sentiment_file)
    
    plt.figure(figsize=(12, 6))
    sns.histplot(df["avg_finbert"], bins=20, kde=True, color="blue", label="FinBERT")
    sns.histplot(df["avg_vader"], bins=20, kde=True, color="orange", label="VADER")
    
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.title("Sentiment Score Distribution Over Time")
    plt.legend()
    
    output_path = "plots/sentiment_distribution.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logging.info(f"✅ Saved sentiment distribution plot at {output_path}.")


def perform_graphing():
    """Runs all graphing functions."""
    plot_sentiment_trends()
    plot_keyword_sentiment()
    plot_financial_vs_climate()
    plot_sentiment_distribution()
    logging.info("✅ Graphing completed successfully.")


if __name__ == "__main__":
    perform_graphing()
