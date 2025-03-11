import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from config import sentiment_flag

ANALYSIS_FOLDER = "analysis/sentiment"
DATA_FOLDER = f"data/{sentiment_flag}_output"


def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        data.columns = data.columns.str.upper()
        logging.info(f"✅ Successfully loaded data from {file_path}.")
        return data
    except Exception as e:
        logging.error(f"❌ Error loading data from {file_path}: {e}")
        raise


def compute_average_sentiment(sentiment_df):
    required_cols = {"DATE", "TICKER", "CATEGORY", "FINBERT_POSITIVE", "FINBERT_NEGATIVE", "VADER_POSITIVE", "VADER_NEGATIVE"}
    if not required_cols.issubset(set(sentiment_df.columns)):
        raise KeyError(f"Missing required columns. Found: {set(sentiment_df.columns)}")

    aggregated_df = (
        sentiment_df.groupby(["DATE", "TICKER", "CATEGORY"])
        .agg(
            AVG_FINBERT=("FINBERT_NEGATIVE", "mean"),
            AVG_VADER=("VADER_NEGATIVE", "mean"),
        )
        .reset_index()
    )

    print(f"✅ Successfully computed average sentiment scores. DataFrame shape: {aggregated_df.shape}")
    return aggregated_df


def plot_sentiment_by_keyword_type_per_ticker(sentiment_df, sentiment_type="FINBERT", output_folder="output/sentiment_tickers"):
    os.makedirs(output_folder, exist_ok=True)

    sentiment_column = f"AVG_{sentiment_type}"

    tickers = sentiment_df["TICKER"].unique()
    categories = ["Financial", "Climate", "Risk"]

    for ticker in tickers:
        plt.figure(figsize=(12, 6))

        for category in categories:
            ticker_data = sentiment_df[(sentiment_df["TICKER"] == ticker) & (sentiment_df["CATEGORY"] == category)]
            if not ticker_data.empty:
                sns.lineplot(data=ticker_data, x="DATE", y=sentiment_column, marker="o", label=category)

        plt.xlabel("Date")
        plt.ylabel(f"Average {sentiment_type.capitalize()} Sentiment Score")
        plt.title(f"{sentiment_type.capitalize()} Sentiment Trends by Keyword Type for {ticker}")
        plt.xticks(rotation=45)
        plt.legend(title="Keyword Category")
        plt.grid()

        output_path = os.path.join(output_folder, f"{sentiment_type.lower()}_{sentiment_flag}_sentiment_{ticker}.png")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        logging.info(f"✅ Saved {sentiment_type} sentiment trends plot for {ticker} at {output_path}.")


def plot_overall_sentiment_by_keyword_type(sentiment_df, sentiment_type="FINBERT", output_folder="output/sentiment_all_tickers"):
    os.makedirs(output_folder, exist_ok=True)

    sentiment_column = f"AVG_{sentiment_type}"

    plt.figure(figsize=(12, 6))

    for category in ["Financial", "Climate", "Risk"]:
        category_data = sentiment_df[sentiment_df["CATEGORY"] == category]
        if not category_data.empty:
            sns.lineplot(data=category_data, x="DATE", y=sentiment_column, marker="o", label=category)

    plt.xlabel("Date")
    plt.ylabel(f"Average {sentiment_type.capitalize()} Sentiment Score")
    plt.title(f"Overall {sentiment_type.capitalize()} Sentiment Trends by Keyword Type")
    plt.xticks(rotation=45)
    plt.legend(title="Keyword Category")
    plt.grid()

    output_path = os.path.join(output_folder, f"overall_{sentiment_type.lower()}_{sentiment_flag}_sentiment.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logging.info(f"✅ Saved overall {sentiment_type} sentiment trends plot at {output_path}.")

def rank_companies_by_sentiment(sentiment_df, output_folder="output/sentiment_all_tickers"):
    
    os.makedirs(output_folder, exist_ok=True)
    output_excel_finbert = os.path.join(output_folder, "ranked_companies_finbert.xlsx")
    output_excel_vader = os.path.join(output_folder, "ranked_companies_vader.xlsx")
    output_csv_finbert = os.path.join(output_folder, "ranked_companies_finbert.csv")
    output_csv_vader = os.path.join(output_folder, "ranked_companies_vader.csv")

    # Compute average sentiment per ticker
    print("🔢 Computing average negative sentiment per company...")
    sentiment_aggregated_df = sentiment_df.groupby("TICKER", as_index=False).agg({
        "AVG_FINBERT": "mean",
        "AVG_VADER": "mean"
    })

    sorted_by_finbert = sentiment_aggregated_df.sort_values(by="AVG_FINBERT", ascending=False).reset_index(drop=True)
    sorted_by_finbert.rename(columns={"AVG_FINBERT": "SORTED_BY_AVG_FINBERT"}, inplace=True)
    
    sorted_by_vader = sentiment_aggregated_df.sort_values(by="AVG_VADER", ascending=False).reset_index(drop=True)
    sorted_by_vader.rename(columns={"AVG_VADER": "SORTED_BY_AVG_VADER"}, inplace=True)

    sorted_by_finbert.to_excel(output_excel_finbert, index=False)
    sorted_by_finbert.to_csv(output_csv_finbert, index=False)
    sorted_by_vader.to_excel(output_excel_vader, index=False)
    sorted_by_vader.to_csv(output_csv_vader, index=False)

def perform_graphing():
    sentiment_file = os.path.join(DATA_FOLDER, f"{sentiment_flag}_sentiment_results_merged.csv")
    sentiment_df = load_data(sentiment_file)

    quarter_to_month = {"1Q": "01", "2Q": "04", "3Q": "07", "4Q": "10"}
    sentiment_df["DATE"] = sentiment_df.apply(
        lambda row: f"{row['YEAR']}-{quarter_to_month.get(row['QUARTER'].upper(), '01')}-01", axis=1
    )
    sentiment_df["DATE"] = pd.to_datetime(sentiment_df["DATE"])

    sentiment_df = compute_average_sentiment(sentiment_df)

    # plot_sentiment_by_keyword_type_per_ticker(sentiment_df, "FINBERT")
    # plot_sentiment_by_keyword_type_per_ticker(sentiment_df, "VADER")

    # plot_overall_sentiment_by_keyword_type(sentiment_df, "FINBERT")
    # plot_overall_sentiment_by_keyword_type(sentiment_df, "VADER")

    rank_companies_by_sentiment(sentiment_df)

    logging.info("✅ Graphing completed successfully.")


if __name__ == "__main__":
    perform_graphing()
