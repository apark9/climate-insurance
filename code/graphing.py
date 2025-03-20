import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from config import sentiment_flag
from scipy.stats import zscore

ANALYSIS_FOLDER = "analysis/sentiment"
DATA_FOLDER = f"data/{sentiment_flag}_output"

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        data.columns = data.columns.str.upper()
        return data
    except Exception as e:
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

def rank_companies_by_sentiment(sentiment_df, output_folder="output/sentiment_all_tickers"):
    
    os.makedirs(output_folder, exist_ok=True)
    output_excel_finbert = os.path.join(output_folder, "ranked_companies_finbert.xlsx")
    output_excel_vader = os.path.join(output_folder, "ranked_companies_vader.xlsx")
    output_csv_finbert = os.path.join(output_folder, "ranked_companies_finbert.csv")
    output_csv_vader = os.path.join(output_folder, "ranked_companies_vader.csv")

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
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    base_data = pd.read_csv(base_file)
    gpt_data = pd.read_csv(gpt_file)

    total_documents = len([f for f in os.listdir(data_folder) if f.endswith(('.txt', '.pdf'))])

    gpt_data.rename(columns={"avg_finbert": "GPT"}, inplace=True)
    merged_data = pd.merge(base_data, gpt_data[["keyword", "GPT"]], on="keyword", how="left")
    merged_data.rename(columns={"avg_finbert": "FinBERT", "avg_vader": "VADER"}, inplace=True)
    merged_data["Frequency (%)"] = (merged_data["count"] / total_documents) * 100
    sentiment_data = merged_data.melt(id_vars=["keyword", "Frequency (%)"], value_vars=["FinBERT", "VADER", "GPT"],
                                      var_name="Model", value_name="Sentiment Score")

    sentiment_data = sentiment_data.sort_values(by="Frequency (%)", ascending=False)

    g = sns.FacetGrid(sentiment_data, col="Model", sharex=True, sharey=True, height=6, aspect=1.2)
    g.map_dataframe(sns.barplot, x="Sentiment Score", y="keyword", hue="Model", palette=["blue", "green", "red"])

    g.set_axis_labels("Sentiment Score", "Keyword")
    g.set_titles("{col_name} Sentiment")
    g.fig.suptitle(title, fontsize=14)
    g.fig.subplots_adjust(top=0.85)

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

def plot_improved_comparison(
    sell_side_base, sell_side_gpt, transcript_base, transcript_gpt, output_file
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    sell_side = pd.read_csv(sell_side_base)
    sell_side_gpt = pd.read_csv(sell_side_gpt)
    transcript = pd.read_csv(transcript_base)
    transcript_gpt = pd.read_csv(transcript_gpt)

    sell_side_gpt.rename(columns={"avg_finbert": "GPT"}, inplace=True)
    sell_side = pd.merge(sell_side, sell_side_gpt[["keyword", "GPT"]], on="keyword", how="left")

    transcript_gpt.rename(columns={"avg_finbert": "GPT"}, inplace=True)
    transcript = pd.merge(transcript, transcript_gpt[["keyword", "GPT"]], on="keyword", how="left")
    sell_side.rename(columns={"avg_finbert": "AVG_FINBERT", "avg_vader": "AVG_VADER"}, inplace=True)
    transcript.rename(columns={"avg_finbert": "AVG_FINBERT", "avg_vader": "AVG_VADER"}, inplace=True)

    sell_side["Source"] = "Sell-Side"
    transcript["Source"] = "Transcripts"

    combined = pd.concat([sell_side, transcript])

    melted = combined.melt(
        id_vars=["keyword", "Source"], value_vars=["AVG_FINBERT", "AVG_VADER", "GPT"],
        var_name="Model", value_name="Sentiment Score"
    )

    melted = melted.groupby(["keyword", "Source", "Model"], as_index=False).mean()

    sentiment_diff = melted.pivot_table(index="keyword", columns="Source", values="Sentiment Score").dropna()
    sentiment_diff["Diff"] = abs(sentiment_diff["Sell-Side"] - sentiment_diff["Transcripts"])
    melted["keyword"] = pd.Categorical(melted["keyword"], categories=sentiment_diff.sort_values("Diff", ascending=False).index)

    plt.figure(figsize=(12, 14))
    sns.pointplot(
        data=melted, x="Sentiment Score", y="keyword",
        hue="Source", dodge=True, markers=["o", "s"], linestyles=""
    )

    plt.xlabel("Sentiment Score")
    plt.ylabel("Keyword")
    plt.title("Sell-Side vs. Transcripts: Keyword Sentiment Analysis")
    plt.legend(title="Source")
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

def plot_standardized_keyword_frequency(
    sell_side_file, sell_side_gpt, transcript_file, transcript_gpt, output_file
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    sell_side = pd.read_csv(sell_side_file)
    sell_side_gpt = pd.read_csv(sell_side_gpt)
    transcript = pd.read_csv(transcript_file)
    transcript_gpt = pd.read_csv(transcript_gpt)

    sell_side_gpt.rename(columns={"avg_finbert": "GPT"}, inplace=True)
    sell_side = pd.merge(sell_side, sell_side_gpt[["keyword", "GPT"]], on="keyword", how="left")

    transcript_gpt.rename(columns={"avg_finbert": "GPT"}, inplace=True)
    transcript = pd.merge(transcript, transcript_gpt[["keyword", "GPT"]], on="keyword", how="left")

    combined = pd.merge(
        sell_side[["keyword", "count"]].rename(columns={"count": "Sell-Side"}),
        transcript[["keyword", "count"]].rename(columns={"count": "Transcripts"}),
        on="keyword",
        how="outer"
    ).fillna(0)

    combined["Sell-Side Z"] = zscore(combined["Sell-Side"])
    combined["Transcripts Z"] = zscore(combined["Transcripts"])

    combined["Diff"] = abs(combined["Sell-Side Z"] - combined["Transcripts Z"])
    combined = combined.sort_values(by="Diff", ascending=False)

    plt.figure(figsize=(12, 8))
    bar_width = 0.4
    keywords = combined["keyword"]
    positions = range(len(keywords))

    plt.barh(positions, combined["Sell-Side Z"], height=bar_width, label="Sell-Side", color="orange", alpha=0.7)
    plt.barh([p + bar_width for p in positions], combined["Transcripts Z"], height=bar_width, label="Transcripts", color="blue", alpha=0.7)

    plt.xlabel("Standardized Keyword Frequency (Z-Score)")
    plt.ylabel("Climate-Related Keywords")
    plt.title("Keyword Frequency Comparison (Z-Score Standardized)")
    plt.yticks([p + bar_width / 2 for p in positions], keywords)
    plt.axvline(x=0, color='black', linestyle="--", linewidth=0.8)
    plt.legend()
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

def plot_top_keywords_by_model(output_folder="output/keyword_sentiment_plots", top_n=5):
    os.makedirs(output_folder, exist_ok=True)

    sentiment_trends_file = os.path.join("analysis/keyword_freq", f"{sentiment_flag}_keyword_sentiment_trends.csv")

    sentiment_df = pd.read_csv(sentiment_trends_file)

    top_keywords = sentiment_df.groupby("keyword")["count"].sum().nlargest(top_n).index.tolist()

    sentiment_df["date"] = sentiment_df["year"].astype(str) + "-" + sentiment_df["quarter"].str.replace("Q", "0")
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"], errors="coerce")

    sentiment_models = {
        "FinBERT": "avg_finbert_negative",
        "VADER": "avg_vader_negative",
        "GPT": "avg_gpt_negative"
    }

    for model_name, column_name in sentiment_models.items():
        plt.figure(figsize=(12, 6))

        for keyword in top_keywords:
            subset = sentiment_df[sentiment_df["keyword"] == keyword]

            sns.lineplot(data=subset, x="date", y=column_name, marker="o", label=f"{keyword}")

        plt.xlabel("Date")
        plt.ylabel("Average Negative Sentiment Score")
        plt.title(f"Sentiment Trends for Top {top_n} Keywords ({model_name})")
        plt.xticks(rotation=45)
        plt.legend(title="Keyword")
        plt.grid()

        output_path = os.path.join(output_folder, f"sentiment_trends_{model_name.lower()}_top_{top_n}_keywords.png")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

def plot_readability_trends(output_folder="output/plots/readability_trends"):
    os.makedirs(output_folder, exist_ok=True)

    sentiment_file = os.path.join(DATA_FOLDER, f"{sentiment_flag}_sentiment_results_merged.csv")
    sentiment_df = pd.read_csv(sentiment_file)
    sentiment_df.columns = sentiment_df.columns.str.lower()

    required_cols = {"year", "quarter", "category", "flesch_reading_ease", "gunning_fog_index", "smog_index", "lexical_density"}

    sentiment_df["time"] = sentiment_df["year"].astype(str) + "-" + sentiment_df["quarter"].str.replace("Q", "0")
    sentiment_df["time"] = pd.to_datetime(sentiment_df["time"], errors="coerce")

    readability_trends = sentiment_df.groupby(["time", "category"], as_index=False).agg(
        flesch_reading_ease=("flesch_reading_ease", "mean"),
        gunning_fog_index=("gunning_fog_index", "mean"),
        smog_index=("smog_index", "mean"),
        lexical_density=("lexical_density", "mean")
    )

    readability_metrics = {
        "Flesch Reading Ease": "flesch_reading_ease",
        "Gunning Fog Index": "gunning_fog_index",
        "SMOG Index": "smog_index",
        "Lexical Density": "lexical_density"
    }

    for title, column in readability_metrics.items():
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=readability_trends, x="time", y=column, hue="category", marker="o")
        
        plt.xlabel("Time (Year-Quarter)")
        plt.ylabel(title)
        plt.title(f"{title} Over Time (Climate vs. Non-Climate)")
        plt.xticks(rotation=45)
        plt.legend(title="Category")
        plt.grid()

        output_path = os.path.join(output_folder, f"readability_trend_{column}.png")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

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

    # plot_keyword_sentiment_analysis(
    #     base_file="analysis/keyword_freq/sell_side_keyword_sentiment.csv",
    #     gpt_file="analysis/keyword_freq/sell_side_keyword_sentiment_gpt.csv",
    #     data_folder="data/sell_side_txt",
    #     title="Sell-Side Reports: Keyword Sentiment Analysis",
    #     output_file="output/plots/sell_side_sentiment_analysis.png"
    # )

    # plot_keyword_sentiment_analysis(
    #     base_file="analysis/keyword_freq/transcripts_keyword_sentiment.csv",
    #     gpt_file="analysis/keyword_freq/transcripts_keyword_sentiment_gpt.csv",
    #     data_folder="data/transcripts",
    #     title="Transcripts: Keyword Sentiment Analysis",
    #     output_file="output/plots/transcript_sentiment_analysis.png"
    # )

    # plot_improved_comparison(
    #     sell_side_base="analysis/keyword_freq/sell_side_keyword_sentiment.csv",
    #     sell_side_gpt="analysis/keyword_freq/sell_side_keyword_sentiment_gpt.csv",
    #     transcript_base="analysis/keyword_freq/transcripts_keyword_sentiment.csv",
    #     transcript_gpt="analysis/keyword_freq/transcripts_keyword_sentiment_gpt.csv",
    #     output_file="output/plots/sell_side_vs_transcripts_better.png"
    # )

    # sell_side_docs = len([f for f in os.listdir("data/sell_side_txt") if f.endswith(('.txt', '.pdf'))])
    # transcript_docs = len([f for f in os.listdir("data/transcripts") if f.endswith(('.txt', '.pdf'))])

    # # Call function to generate improved keyword frequency comparison
    # plot_standardized_keyword_frequency(
    #     sell_side_file="analysis/keyword_freq/sell_side_keyword_sentiment.csv",
    #     sell_side_gpt="analysis/keyword_freq/sell_side_keyword_sentiment_gpt.csv",
    #     transcript_file="analysis/keyword_freq/transcripts_keyword_sentiment.csv",
    #     transcript_gpt="analysis/keyword_freq/transcripts_keyword_sentiment_gpt.csv",
    #     output_file="output/plots/standardized_keyword_frequency.png"
    # )

    # plot_top_keywords_by_model(output_folder="output/plots/keyword_sentiment_plots", top_n=3)
    plot_readability_trends()


if __name__ == "__main__":
    perform_graphing()
