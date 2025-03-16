import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Load the dataset
file_path = "data/emdat.csv"
df = pd.read_csv(file_path, dtype=str)
output_dir = "output/plots/climate"

# List of states to filter
states = ["California", "Florida", "Texas"]

# Define a function to convert month to quarter
def get_quarter(month):
    month = int(month)
    if month in [1, 2, 3]:
        return "1Q"
    elif month in [4, 5, 6]:
        return "2Q"
    elif month in [7, 8, 9]:
        return "3Q"
    elif month in [10, 11, 12]:
        return "4Q"
    return None

def filter_dataset():
    # Print available column names for debugging
    print("Available Columns in Dataset:", df.columns.tolist())

    # Strip column names of unexpected spaces
    df.columns = df.columns.str.strip()

    # Identify the correct column name for damages
    possible_damage_cols = [
        "Total Damage, Adjusted ('000 US$)",
        "Total Damage, Adjusted (000 US$)",
        "Total Damage Adjusted ('000 US$)",
    ]

    damage_col = next((col for col in possible_damage_cols if col in df.columns), None)
    
    if not damage_col:
        raise KeyError("Could not find the 'Total Damage, Adjusted' column in dataset!")

    for state in states:
        # Filter data for the state
        state_df = df[df['Location'].str.contains(state, na=False, case=False)]
        
        # Keep necessary columns
        state_df = state_df[['End Year', 'End Month', damage_col]].copy()
        
        # Convert columns to appropriate types
        state_df.dropna(subset=['End Year', 'End Month'], inplace=True)
        state_df['End Year'] = state_df['End Year'].astype(int)
        state_df['End Month'] = state_df['End Month'].astype(int)
        state_df[damage_col] = pd.to_numeric(state_df[damage_col], errors='coerce')  # Convert damages to numeric
        
        # Create Quarter column
        state_df['Quarter'] = state_df['End Month'].apply(get_quarter)
        
        # Aggregate damages by Year & Quarter (summing across events)
        state_df = state_df.groupby(['End Year', 'Quarter'], as_index=False)[damage_col].sum()
        
        # Rename columns
        state_df.columns = ['Year', 'Quarter', 'Damages']
        
        # Save the processed data
        output_path = f"data/climate/emdat_climate_shocks_{state}.csv"
        state_df.to_csv(output_path, index=False)

    print("Aggregated files have been successfully created for California, Florida, and Texas.")

def process_sentiment_data():
    """
    Processes sentiment datasets for specified companies in California, Florida, and Texas,
    aggregating average negative sentiment scores per quarter.
    Includes keyword and category in the saved dataset.
    """

    # Define file paths
    vader_finbert_file = "data/transcripts_output/transcripts_sentiment_results_merged.csv"
    gpt_file = "data/transcripts_output_gpt/transcripts_sentiment_results_merged_gpt.csv"
    
    # Define companies by state
    state_companies = {
        "California": {"AIG", "ALL", "BRK", "CB", "MCY", "TRV", "ZURN"},
        "Florida": {"AIG", "AIZ", "ALL", "BRK", "CB", "PGR", "TRV", "UVE", "ZURN"},
        "Texas": {"AIG", "ALL", "BRK", "CB", "PGR", "TRV", "ZURN"}
    }

    # Columns to keep
    sentiment_columns = ["Ticker", "Year", "Quarter", "finbert_negative", "vader_negative", "keyword", "category"]
    gpt_column_rename = {"finbert_negative": "gpt_negative"}

    # Load FinBERT & Vader dataset
    try:
        finbert_vader_df = pd.read_csv(vader_finbert_file, dtype=str)
        for col in ["finbert_negative", "vader_negative"]:
            finbert_vader_df[col] = pd.to_numeric(finbert_vader_df[col], errors="coerce")
        finbert_vader_df["Quarter"] = finbert_vader_df["Quarter"].str.replace("Q", "") + "Q"
        finbert_vader_df = finbert_vader_df[sentiment_columns]
    except FileNotFoundError:
        finbert_vader_df = pd.DataFrame(columns=sentiment_columns)

    # Load GPT dataset
    try:
        gpt_df = pd.read_csv(gpt_file, dtype=str)
        if "finbert_negative" in gpt_df.columns:
            gpt_df.rename(columns=gpt_column_rename, inplace=True)
            gpt_df["gpt_negative"] = pd.to_numeric(gpt_df["gpt_negative"], errors="coerce")
        gpt_df["Quarter"] = gpt_df["Quarter"].str.extract(r"(\d)").astype(str) + "Q"
        gpt_df = gpt_df[["Ticker", "Year", "Quarter", "gpt_negative"]]
    except FileNotFoundError:
        gpt_df = pd.DataFrame(columns=["Ticker", "Year", "Quarter", "gpt_negative"])

    # Merge sentiment datasets
    combined_df = pd.merge(finbert_vader_df, gpt_df, on=["Ticker", "Year", "Quarter"], how="outer")

    # Convert Year to integer for consistency
    combined_df["Year"] = combined_df["Year"].astype(int)

    # Process data for each state
    for state, companies in state_companies.items():
        state_df = combined_df[combined_df["Ticker"].isin(companies)].copy()

        # Aggregate sentiment scores & count keyword frequency
        state_df = state_df.groupby(["Year", "Quarter", "Ticker", "keyword", "category"], as_index=False).agg({
            "finbert_negative": "mean",
            "vader_negative": "mean",
            "gpt_negative": "mean"
        })

        # Save to CSV including keyword and category
        output_path = f"data/climate/sentiment_results_{state}.csv"
        state_df.to_csv(output_path, index=False)

    print("Sentiment datasets successfully created for California, Florida, and Texas.")

def plot_sentiment_vs_damages(state):
    """
    Plots overall sentiment scores vs. damages for a given state,
    using quarter as the x-axis instead of aggregating by year.
    """
    sentiment_file = f"data/climate/sentiment_results_{state}.csv"
    damage_file = f"data/climate/emdat_climate_shocks_{state}.csv"

    try:
        sentiment_df = pd.read_csv(sentiment_file)
        damage_df = pd.read_csv(damage_file)

        sentiment_df["Time"] = sentiment_df["Year"].astype(str) + "-" + sentiment_df["Quarter"]
        damage_df["Time"] = damage_df["Year"].astype(str) + "-" + damage_df["Quarter"]

        # Merge datasets
        merged_df = sentiment_df.groupby("Time", as_index=False).mean(numeric_only=True).merge(
            damage_df, on="Time", how="left")

        plt.figure(figsize=(12, 6))
        ax1 = sns.lineplot(data=merged_df, x="Time", y="finbert_negative", label="FinBERT Sentiment", color="blue")
        sns.lineplot(data=merged_df, x="Time", y="vader_negative", label="VADER Sentiment", color="green")
        sns.lineplot(data=merged_df, x="Time", y="gpt_negative", label="GPT Sentiment", color="red")
        ax2 = ax1.twinx()
        sns.lineplot(data=merged_df, x="Time", y="Damages", label="Damages ($ millions)", color="black", ax=ax2, linestyle="dashed")

        # Format y-axis for damages
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}'))

        ax1.set_ylabel("Sentiment Score")
        ax2.set_ylabel("Damages ($ millions)")

        # Improve x-axis readability
        plt.xticks(range(0, len(merged_df["Time"]), 4), merged_df["Time"][::4], rotation=45)
        plt.xlabel("Time (Year-Quarter)")
        plt.title(f"Overall Sentiment vs. Damages Over Time - {state}")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

        # Save plot
        plot_path = f"{output_dir}/sentiment_vs_damages_{state}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    except FileNotFoundError:
        print(f"Data missing for {state}, skipping plot.")


def plot_climate_sentiment_vs_damages(state):
    """
    Plots climate-related sentiment vs. damages for a given state,
    using the category column instead of hardcoded keywords.
    """
    sentiment_file = f"data/climate/sentiment_results_{state}.csv"
    damage_file = f"data/climate/emdat_climate_shocks_{state}.csv"

    try:
        sentiment_df = pd.read_csv(sentiment_file)
        damage_df = pd.read_csv(damage_file)

        sentiment_df["Time"] = sentiment_df["Year"].astype(str) + "-" + sentiment_df["Quarter"]
        damage_df["Time"] = damage_df["Year"].astype(str) + "-" + damage_df["Quarter"]

        # Filter for climate-related categories
        climate_df = sentiment_df[sentiment_df["category"] == "Climate"]

        climate_merged_df = climate_df.groupby("Time", as_index=False).mean(numeric_only=True).merge(
            damage_df, on="Time", how="left")

        plt.figure(figsize=(12, 6))
        ax1 = sns.lineplot(data=climate_merged_df, x="Time", y="finbert_negative", label="FinBERT Sentiment", color="blue")
        sns.lineplot(data=climate_merged_df, x="Time", y="vader_negative", label="VADER Sentiment", color="green")
        sns.lineplot(data=climate_merged_df, x="Time", y="gpt_negative", label="GPT Sentiment", color="red")
        ax2 = ax1.twinx()
        sns.lineplot(data=climate_merged_df, x="Time", y="Damages", label="Damages ($ millions)", color="black", ax=ax2, linestyle="dashed")

        # Format y-axis for damages
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}'))

        ax1.set_ylabel("Climate Sentiment Score")
        ax2.set_ylabel("Damages ($ millions)")

        # Improve x-axis readability
        plt.xticks(range(0, len(climate_merged_df["Time"]), 4), climate_merged_df["Time"][::4], rotation=45)
        plt.xlabel("Time (Year-Quarter)")
        plt.title(f"Climate Sentiment vs. Damages Over Time - {state}")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

        # Save plot
        plot_path = f"{output_dir}/climate_sentiment_vs_damages_{state}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    except FileNotFoundError:
        print(f"Data missing for {state}, skipping plot.")


def plot_climate_keyword_freq_vs_damages(state):
    """
    Plots climate keyword frequency vs. damages for a given state,
    using the category column instead of hardcoded keywords.
    """
    sentiment_file = f"data/climate/sentiment_results_{state}.csv"
    damage_file = f"data/climate/emdat_climate_shocks_{state}.csv"

    try:
        sentiment_df = pd.read_csv(sentiment_file)
        damage_df = pd.read_csv(damage_file)

        sentiment_df["Time"] = sentiment_df["Year"].astype(str) + "-" + sentiment_df["Quarter"]
        damage_df["Time"] = damage_df["Year"].astype(str) + "-" + damage_df["Quarter"]

        # Filter for climate-related categories
        keyword_freq = sentiment_df[sentiment_df["category"] == "Climate"]
        keyword_freq = keyword_freq.groupby("Time").size().reset_index(name="Keyword Frequency")

        keyword_freq_merged = keyword_freq.merge(damage_df, on="Time", how="left")

        plt.figure(figsize=(12, 6))
        ax1 = sns.lineplot(data=keyword_freq_merged, x="Time", y="Keyword Frequency", label="Climate Keyword Frequency", color="blue")
        ax2 = ax1.twinx()
        sns.lineplot(data=keyword_freq_merged, x="Time", y="Damages", label="Damages ($ millions)", color="black", ax=ax2, linestyle="dashed")

        # Format y-axis for damages
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}'))

        ax1.set_ylabel("Climate Keyword Frequency")
        ax2.set_ylabel("Damages ($ millions)")

        # Improve x-axis readability
        plt.xticks(range(0, len(keyword_freq_merged["Time"]), 4), keyword_freq_merged["Time"][::4], rotation=45)
        plt.xlabel("Time (Year-Quarter)")
        plt.title(f"Climate Keyword Frequency vs. Damages Over Time - {state}")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

        # Save plot
        plot_path = f"{output_dir}/climate_keyword_freq_vs_damages_{state}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    except FileNotFoundError:
        print(f"Data missing for {state}, skipping plot.")

def perform_graphing():
    states = ["California", "Florida", "Texas"]
    for state in states:
        plot_sentiment_vs_damages(state)
        plot_climate_sentiment_vs_damages(state)
        plot_climate_keyword_freq_vs_damages(state)

