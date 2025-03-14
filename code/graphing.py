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

def plot_keyword_sentiment_analysis(base_file, gpt_file, data_folder, title, output_file):
    """
    Creates separate keyword sentiment analysis plots for Sell-Side Reports and Transcripts.
    Uses a Facet Grid to split results across FinBERT, VADER, and GPT.

    Parameters:
        base_file (str): Path to the base sentiment file (FinBERT, VADER).
        gpt_file (str): Path to the GPT sentiment file.
        data_folder (str): Folder containing transcript or sell-side files.
        title (str): Title for the plot (Sell-Side Reports or Transcripts).
        output_file (str): Output file path for the saved graph.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load datasets
    base_data = pd.read_csv(base_file)
    gpt_data = pd.read_csv(gpt_file)

    # Count total documents in the respective folder
    total_documents = len([f for f in os.listdir(data_folder) if f.endswith(('.txt', '.pdf'))])
    if total_documents == 0:
        logging.warning(f"⚠️ No documents found in {data_folder}.")
        return

    # Rename GPT column
    gpt_data.rename(columns={"avg_finbert": "GPT"}, inplace=True)

    # Merge datasets on 'keyword'
    merged_data = pd.merge(base_data, gpt_data[["keyword", "GPT"]], on="keyword", how="left")
    merged_data.rename(columns={"avg_finbert": "FinBERT", "avg_vader": "VADER"}, inplace=True)

    # Calculate frequency percentage
    merged_data["Frequency (%)"] = (merged_data["count"] / total_documents) * 100

    # Reshape for Seaborn plotting
    sentiment_data = merged_data.melt(id_vars=["keyword", "Frequency (%)"], value_vars=["FinBERT", "VADER", "GPT"],
                                      var_name="Model", value_name="Sentiment Score")

    # Sort keywords by frequency for better readability
    sentiment_data = sentiment_data.sort_values(by="Frequency (%)", ascending=False)

    # Set up Facet Grid
    g = sns.FacetGrid(sentiment_data, col="Model", sharex=True, sharey=True, height=6, aspect=1.2)
    g.map_dataframe(sns.barplot, x="Sentiment Score", y="keyword", hue="Model", palette=["blue", "green", "red"])

    # Adjust labels and formatting
    g.set_axis_labels("Sentiment Score", "Keyword")
    g.set_titles("{col_name} Sentiment")
    g.fig.suptitle(title, fontsize=14)
    g.fig.subplots_adjust(top=0.85)

    # Save plot
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
    logging.info(f"✅ Saved {title} analysis to {output_file}.")

def plot_improved_comparison(
    sell_side_base, sell_side_gpt, transcript_base, transcript_gpt, output_file
):
    """
    Creates a clearer Sell-Side vs. Transcripts comparison by stacking sentiment scores vertically.
    
    Parameters:
        sell_side_base (str): CSV file for Sell-Side FinBERT/VADER.
        sell_side_gpt (str): CSV file for Sell-Side GPT.
        transcript_base (str): CSV file for Transcripts FinBERT/VADER.
        transcript_gpt (str): CSV file for Transcripts GPT.
        output_file (str): Where to save the plot.
    """
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load Data
    sell_side = pd.read_csv(sell_side_base)
    sell_side_gpt = pd.read_csv(sell_side_gpt)
    transcript = pd.read_csv(transcript_base)
    transcript_gpt = pd.read_csv(transcript_gpt)

    # Rename GPT column and merge
    sell_side_gpt.rename(columns={"avg_finbert": "GPT"}, inplace=True)
    sell_side = pd.merge(sell_side, sell_side_gpt[["keyword", "GPT"]], on="keyword", how="left")

    transcript_gpt.rename(columns={"avg_finbert": "GPT"}, inplace=True)
    transcript = pd.merge(transcript, transcript_gpt[["keyword", "GPT"]], on="keyword", how="left")

    # Rename sentiment columns
    sell_side.rename(columns={"avg_finbert": "AVG_FINBERT", "avg_vader": "AVG_VADER"}, inplace=True)
    transcript.rename(columns={"avg_finbert": "AVG_FINBERT", "avg_vader": "AVG_VADER"}, inplace=True)

    # Add Source Labels
    sell_side["Source"] = "Sell-Side"
    transcript["Source"] = "Transcripts"

    # Merge both datasets
    combined = pd.concat([sell_side, transcript])

    # Reshape for Seaborn
    melted = combined.melt(
        id_vars=["keyword", "Source"], value_vars=["AVG_FINBERT", "AVG_VADER", "GPT"],
        var_name="Model", value_name="Sentiment Score"
    )

    # Fix duplicate keyword-source pairs by averaging
    melted = melted.groupby(["keyword", "Source", "Model"], as_index=False).mean()

    # Sort by the largest sentiment gap
    sentiment_diff = melted.pivot_table(index="keyword", columns="Source", values="Sentiment Score").dropna()
    sentiment_diff["Diff"] = abs(sentiment_diff["Sell-Side"] - sentiment_diff["Transcripts"])
    melted["keyword"] = pd.Categorical(melted["keyword"], categories=sentiment_diff.sort_values("Diff", ascending=False).index)

    # Plot
    plt.figure(figsize=(12, 14))
    sns.pointplot(
        data=melted, x="Sentiment Score", y="keyword",
        hue="Source", dodge=True, markers=["o", "s"], linestyles=""
    )

    # Labels and Formatting
    plt.xlabel("Sentiment Score")
    plt.ylabel("Keyword")
    plt.title("Sell-Side vs. Transcripts: Keyword Sentiment Analysis")
    plt.legend(title="Source")
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
    logging.info(f"✅ Saved improved sentiment analysis to {output_file}.")

def perform_graphing():
    # sentiment_file = os.path.join(DATA_FOLDER, f"{sentiment_flag}_sentiment_results_merged.csv")
    # sentiment_df = load_data(sentiment_file)

    # quarter_to_month = {"1Q": "01", "2Q": "04", "3Q": "07", "4Q": "10"}
    # sentiment_df["DATE"] = sentiment_df.apply(
    #     lambda row: f"{row['YEAR']}-{quarter_to_month.get(row['QUARTER'].upper(), '01')}-01", axis=1
    # )
    # sentiment_df["DATE"] = pd.to_datetime(sentiment_df["DATE"])

    # sentiment_df = compute_average_sentiment(sentiment_df)

    # plot_sentiment_by_keyword_type_per_ticker(sentiment_df, "FINBERT")
    # plot_sentiment_by_keyword_type_per_ticker(sentiment_df, "VADER")

    # plot_overall_sentiment_by_keyword_type(sentiment_df, "FINBERT")
    # plot_overall_sentiment_by_keyword_type(sentiment_df, "VADER")

    # rank_companies_by_sentiment(sentiment_df)

    plot_keyword_sentiment_analysis(
        base_file="analysis/keyword_freq/sell_side_keyword_sentiment.csv",
        gpt_file="analysis/keyword_freq/sell_side_keyword_sentiment_gpt.csv",
        data_folder="data/sell_side_txt",
        title="Sell-Side Reports: Keyword Sentiment Analysis",
        output_file="output/plots/sell_side_sentiment_analysis.png"
    )

    plot_keyword_sentiment_analysis(
        base_file="analysis/keyword_freq/transcripts_keyword_sentiment.csv",
        gpt_file="analysis/keyword_freq/transcripts_keyword_sentiment_gpt.csv",
        data_folder="data/transcripts",
        title="Transcripts: Keyword Sentiment Analysis",
        output_file="output/plots/transcript_sentiment_analysis.png"
    )

    plot_improved_comparison(
        sell_side_base="analysis/keyword_freq/sell_side_keyword_sentiment.csv",
        sell_side_gpt="analysis/keyword_freq/sell_side_keyword_sentiment_gpt.csv",
        transcript_base="analysis/keyword_freq/transcripts_keyword_sentiment.csv",
        transcript_gpt="analysis/keyword_freq/transcripts_keyword_sentiment_gpt.csv",
        output_file="output/plots/sell_side_vs_transcripts_better.png"
    )

    logging.info("✅ Graphing completed successfully.")


if __name__ == "__main__":
    perform_graphing()
